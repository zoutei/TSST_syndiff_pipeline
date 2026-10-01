# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""Headless emergency Drive upload from Colab when the login-host supervisor is absent."""
from __future__ import annotations

import shutil
import sys
import time
import zipfile
from pathlib import Path

DEFAULT_PARENT_FOLDER_ID = "1dkh6uz7BwuhX-l6iz5ZVkSis1FkDG6Yr"
DEFAULT_SCOPES = ["https://www.googleapis.com/auth/drive.file"]
# Mirror scripts/colab_job.py ESSENTIAL_ARTIFACTS (pack cannot import colab_job).
ESSENTIAL_ARTIFACTS_FALLBACK = (
    "status.json", "train.log", "history.jsonl", "history.json", "fit_meta.json",
    "params_latest.npz", "params_stage3.npz", "params_stage2.npz", "params_stage1.npz",
    "params_stage0.npz", "training_state_latest.npz", "telemetry.jsonl", "runner.log",
    "thread_env.txt", "level1_audit_stage1.csv", "level1_audit_stage2.csv", "level1_audit_stage3.csv",
    "flux_solved.npz",
)
PRIVATE_ARTIFACT_DIRS = (".gdrive_oauth",)
UPLOAD_RETRY_BACKOFF_S = 2.0


def is_private_artifact(rel: str) -> bool:
    name = Path(rel).name
    # These are VM bootstrap/control files, not fit outputs.  In particular,
    # never mirror the Discord webhook onto shared login-host storage.
    if name == ".discord_webhook":
        return True
    if name in {".provision_launch.token", ".provision_worker.pid"}:
        return True
    if name.startswith(".phase_") and name.endswith(".done"):
        return True
    return any(rel == name or rel.startswith(f"{name}/") for name in PRIVATE_ARTIFACT_DIRS)


def emergency_zip_name(run_id: str) -> str:
    return f"{run_id}_emergency_artifacts.zip"


def oauth_dir(job_dir: Path) -> Path:
    return job_dir / PRIVATE_ARTIFACT_DIRS[0]


def artifact_rel_paths(manifest: dict[str, object]) -> list[str]:
    if manifest:
        return sorted(manifest.keys())
    return list(ESSENTIAL_ARTIFACTS_FALLBACK)


def build_emergency_zip(
    job_dir: Path,
    run_id: str,
    manifest: dict[str, object],
) -> Path:
    job_dir = job_dir.resolve()
    zip_path = job_dir / emergency_zip_name(run_id)
    if zip_path.is_file():
        zip_path.unlink()
    members = set(artifact_rel_paths(manifest))
    manifest_path = job_dir / "artifact_manifest.json"
    if manifest_path.is_file():
        members.add("artifact_manifest.json")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        for rel in sorted(members):
            if rel == zip_path.name or ".tmp." in rel:
                continue
            if is_private_artifact(rel):
                continue
            path = job_dir / rel
            if not path.is_file():
                continue
            archive.write(path, rel)
    return zip_path


def build_drive_service(token_file: Path, credentials_file: Path):
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build

    if not token_file.is_file():
        raise FileNotFoundError(f"OAuth token not found: {token_file}")
    creds = Credentials.from_authorized_user_file(str(token_file), DEFAULT_SCOPES)
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
    if not creds.valid:
        if not credentials_file.is_file():
            raise RuntimeError(f"OAuth token invalid and no credentials file: {credentials_file}")
        raise RuntimeError("OAuth token invalid and could not refresh")
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def _find_folder(service, name: str, parent_id: str) -> str | None:
    query = (
        f"name = '{name}' and '{parent_id}' in parents and "
        "mimeType = 'application/vnd.google-apps.folder' and trashed = false"
    )
    result = service.files().list(q=query, spaces="drive", fields="files(id, name)", pageSize=10).execute()
    files = result.get("files", [])
    return files[0]["id"] if files else None


def _create_folder(service, name: str, parent_id: str) -> str:
    metadata = {
        "name": name,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [parent_id],
    }
    folder = service.files().create(body=metadata, fields="id, name").execute()
    return folder["id"]


def ensure_subfolder(service, parent_id: str, subfolder_name: str) -> str:
    folder_id = parent_id
    for component in (part for part in subfolder_name.split("/") if part):
        child_id = _find_folder(service, component, folder_id)
        if child_id is None:
            child_id = _create_folder(service, component, folder_id)
        folder_id = child_id
    return folder_id


def _find_file_in_folder(service, name: str, folder_id: str) -> str | None:
    query = (
        f"name = '{name}' and '{folder_id}' in parents and "
        "mimeType != 'application/vnd.google-apps.folder' and trashed = false"
    )
    result = service.files().list(q=query, spaces="drive", fields="files(id, name)", pageSize=10).execute()
    files = result.get("files", [])
    return files[0]["id"] if files else None


def upload_zip_to_drive(
    zip_path: Path,
    run_id: str,
    token_file: Path,
    credentials_file: Path,
    *,
    parent_folder_id: str = DEFAULT_PARENT_FOLDER_ID,
    max_attempts: int = 3,
) -> tuple[str | None, str | None]:
    from googleapiclient.http import MediaFileUpload

    zip_path = zip_path.resolve()
    if not zip_path.is_file():
        return None, f"emergency zip missing: {zip_path}"
    if not token_file.is_file():
        return None, f"OAuth token not found: {token_file}"
    subfolder = f"syndiff/runs/{run_id}"
    file_name = zip_path.name
    last_error = "upload not attempted"
    for attempt in range(max_attempts):
        try:
            service = build_drive_service(token_file, credentials_file)
            target_folder_id = ensure_subfolder(service, parent_folder_id, subfolder)
            existing_id = _find_file_in_folder(service, file_name, target_folder_id)
            media = MediaFileUpload(str(zip_path), resumable=True)
            if existing_id:
                request = service.files().update(
                    fileId=existing_id,
                    media_body=media,
                    fields="id, name, size",
                )
            else:
                request = service.files().create(
                    body={"name": file_name, "parents": [target_folder_id]},
                    media_body=media,
                    fields="id, name, size",
                )
            response = None
            while response is None:
                _status, response = request.next_chunk()
            file_id = response.get("id")
            if file_id:
                return str(file_id), None
            last_error = "Drive API returned no file id"
        except Exception as exc:
            last_error = str(exc)
        if attempt + 1 < max_attempts:
            time.sleep(UPLOAD_RETRY_BACKOFF_S * (attempt + 1))
    return None, last_error


def cleanup_oauth_dir(job_dir: Path) -> None:
    oauth_path = oauth_dir(job_dir)
    if oauth_path.is_dir():
        shutil.rmtree(oauth_path, ignore_errors=True)


def colab_unassign() -> bool:
    try:
        from google.colab import runtime  # noqa: PLC0415
        runtime.unassign()
        return True
    except Exception as exc:
        print(f"colab unassign skipped or failed: {exc}", file=sys.stderr, flush=True)
        return False


def emergency_upload_and_release(
    job_dir: Path,
    run_id: str,
    manifest: dict[str, object],
    *,
    max_upload_attempts: int = 3,
) -> tuple[str | None, str | None]:
    """Zip artifacts, upload to Drive, cleanup token, unassign on success."""
    token_file = oauth_dir(job_dir) / "token.json"
    credentials_file = oauth_dir(job_dir) / "credentials.json"
    if not token_file.is_file():
        return None, "shipped OAuth token missing; emergency upload skipped"
    zip_path = build_emergency_zip(job_dir, run_id, manifest)
    file_id, error = upload_zip_to_drive(
        zip_path,
        run_id,
        token_file,
        credentials_file,
        max_attempts=max_upload_attempts,
    )
    if file_id:
        cleanup_oauth_dir(job_dir)
        colab_unassign()
        return file_id, None
    return None, error
