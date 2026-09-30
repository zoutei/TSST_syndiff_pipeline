# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
#!/usr/bin/env python3
"""Upload Colab training bundles to Google Drive.

Uses OAuth (``drive.file`` scope) and uploads into a ``syndiff`` subfolder of the
configured parent Drive folder. Re-uploading the same filename updates the
existing Drive file when this app created it previously (stable file ID for
``gdown``).

First run prints a Google login URL in the terminal. Open it in your browser,
approve access, then paste the redirect URL (or just the ``code`` value) back
into the terminal. The session is saved to ``~/token.json`` for later uploads.

Example::

    python dev/forward_epsf_wcs/scripts/upload_to_gdrive.py \\
        dev/forward_epsf_wcs/output/colab/colab_fullccd_mag710_irreg_295.zip
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from urllib.parse import parse_qs, urlparse

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

DEFAULT_CREDENTIALS_FILE = os.path.expanduser("~/google_oauth_credentials.json")
DEFAULT_TOKEN_FILE = os.path.expanduser("~/token.json")
DEFAULT_PARENT_FOLDER_ID = "1dkh6uz7BwuhX-l6iz5ZVkSis1FkDG6Yr"
DEFAULT_SUBFOLDER_NAME = "syndiff"
DEFAULT_BUNDLE_ZIP = (
    Path(__file__).resolve().parents[1] / "output/colab/colab_fullccd_mag710_irreg_295.zip"
)
# Restricted scope: only files created/opened by this app.
SCOPES = ["https://www.googleapis.com/auth/drive.file"]


def _parse_auth_code(response: str) -> str | None:
    """Extract authorization code from a redirect URL or bare paste."""
    response = response.strip().strip("'\"")
    if not response:
        return None
    if response.startswith("http"):
        parsed = urlparse(response)
        code = (parse_qs(parsed.query).get("code") or [None])[0]
        return code
    # User may paste ``code&scope=...`` without the leading URL.
    return response.split("&", 1)[0]


def _authenticate_installed_app(flow: InstalledAppFlow) -> Credentials:
    """Interactive OAuth for cluster login nodes (URL in terminal, paste back)."""
    if hasattr(flow, "run_console"):
        return flow.run_console()

    flow.redirect_uri = "http://localhost"
    auth_url, _ = flow.authorization_url(prompt="consent", access_type="offline")

    print()
    print("=" * 72)
    print("Google Drive login (one-time)")
    print("=" * 72)
    print()
    print("Step 1 — open this URL in your browser:")
    print()
    print(auth_url)
    print()
    print("Step 2 — sign in and click Allow.")
    print()
    print(
        "Step 3 — the browser redirects to localhost and may show an error page.\n"
        "         That is normal. Copy the FULL URL from the address bar, e.g.:"
    )
    print("         http://localhost/?code=4/0A...&scope=...")
    print()
    print("Step 4 — paste the full redirect URL or just the code value below.")
    print()

    while True:
        response = input("Paste redirect URL or code: ").strip()
        code = _parse_auth_code(response)
        if not code:
            print("No code found — try again, or Ctrl-C to cancel.")
            continue
        try:
            # Always exchange via code= (not authorization_response=) so we avoid
            # oauthlib's insecure_transport error on http://localhost redirects.
            flow.fetch_token(code=code)
            break
        except Exception as exc:
            print(f"Could not exchange code ({exc}). Try copying the URL again.")
            continue

    print()
    print(f"Login OK — saving token to {DEFAULT_TOKEN_FILE}")
    print()
    return flow.credentials


def get_gdrive_service(
    credentials_file: str = DEFAULT_CREDENTIALS_FILE,
    token_file: str = DEFAULT_TOKEN_FILE,
    *,
    force_login: bool = False,
):
    creds = None
    if not force_login and os.path.exists(token_file):
        creds = Credentials.from_authorized_user_file(token_file, SCOPES)

    if force_login or not creds or not creds.valid:
        if not force_login and creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(credentials_file):
                raise FileNotFoundError(
                    f"OAuth client secrets not found: {credentials_file}"
                )
            flow = InstalledAppFlow.from_client_secrets_file(credentials_file, SCOPES)
            creds = _authenticate_installed_app(flow)

        with open(token_file, "w", encoding="utf-8") as token:
            token.write(creds.to_json())
        os.chmod(token_file, 0o600)

    return build("drive", "v3", credentials=creds, cache_discovery=False)


def find_folder(service, name: str, parent_id: str) -> str | None:
    query = (
        f"name = '{name}' and '{parent_id}' in parents and "
        "mimeType = 'application/vnd.google-apps.folder' and trashed = false"
    )
    result = (
        service.files()
        .list(q=query, spaces="drive", fields="files(id, name)", pageSize=10)
        .execute()
    )
    files = result.get("files", [])
    return files[0]["id"] if files else None


def create_folder(service, name: str, parent_id: str) -> str:
    metadata = {
        "name": name,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [parent_id],
    }
    folder = service.files().create(body=metadata, fields="id, name").execute()
    return folder["id"]


def ensure_subfolder(service, parent_id: str, subfolder_name: str) -> str:
    """Ensure a slash-separated folder path, preserving stable folder IDs."""
    folder_id = parent_id
    for component in (part for part in subfolder_name.split("/") if part):
        child_id = find_folder(service, component, folder_id)
        if child_id is None:
            child_id = create_folder(service, component, folder_id)
            print(f"Created folder '{component}' (id={child_id})")
        else:
            print(f"Using existing folder '{component}' (id={child_id})")
        folder_id = child_id
    return folder_id


def find_file_in_folder(service, name: str, folder_id: str) -> str | None:
    query = (
        f"name = '{name}' and '{folder_id}' in parents and "
        "mimeType != 'application/vnd.google-apps.folder' and trashed = false"
    )
    result = (
        service.files()
        .list(q=query, spaces="drive", fields="files(id, name)", pageSize=10)
        .execute()
    )
    files = result.get("files", [])
    return files[0]["id"] if files else None


def upload_file_to_gdrive(
    local_file_path: str | Path,
    *,
    credentials_file: str = DEFAULT_CREDENTIALS_FILE,
    token_file: str = DEFAULT_TOKEN_FILE,
    parent_folder_id: str = DEFAULT_PARENT_FOLDER_ID,
    subfolder_name: str = DEFAULT_SUBFOLDER_NAME,
    force_login: bool = False,
) -> dict:
    local_path = Path(local_file_path).expanduser().resolve()
    if not local_path.is_file():
        raise FileNotFoundError(f"Local file does not exist: {local_path}")

    service = get_gdrive_service(credentials_file, token_file, force_login=force_login)
    target_folder_id = ensure_subfolder(service, parent_folder_id, subfolder_name)

    file_name = local_path.name
    existing_id = find_file_in_folder(service, file_name, target_folder_id)
    media = MediaFileUpload(str(local_path), resumable=True)

    if existing_id:
        print(f"Updating existing Drive file '{file_name}' (id={existing_id})")
        request = service.files().update(
            fileId=existing_id,
            media_body=media,
            fields="id, name, size, webViewLink",
        )
    else:
        print(f"Uploading new file '{file_name}' to '{subfolder_name}/'")
        request = service.files().create(
            body={"name": file_name, "parents": [target_folder_id]},
            media_body=media,
            fields="id, name, size, webViewLink",
        )

    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            print(f"  uploaded {int(status.progress() * 100)}%")

    size_mb = int(response.get("size", 0)) / (1024 * 1024)
    print(
        f"Upload complete: {response['name']} "
        f"(id={response['id']}, size={size_mb:.1f} MiB)"
    )
    print(f"gdown: gdown {response['id']} -O {file_name}")
    if response.get("webViewLink"):
        print(f"Drive link: {response['webViewLink']}")
    return response


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload a Colab bundle zip to Google Drive (syndiff subfolder).",
    )
    parser.add_argument(
        "local_file",
        nargs="?",
        default=str(DEFAULT_BUNDLE_ZIP),
        help=f"Local zip to upload (default: {DEFAULT_BUNDLE_ZIP})",
    )
    parser.add_argument(
        "--credentials",
        default=DEFAULT_CREDENTIALS_FILE,
        help=f"OAuth client secrets JSON (default: {DEFAULT_CREDENTIALS_FILE})",
    )
    parser.add_argument(
        "--token",
        default=DEFAULT_TOKEN_FILE,
        help=f"Saved OAuth token JSON (default: {DEFAULT_TOKEN_FILE})",
    )
    parser.add_argument(
        "--parent-folder-id",
        default=DEFAULT_PARENT_FOLDER_ID,
        help="Parent Drive folder ID (syndiff is created inside this)",
    )
    parser.add_argument(
        "--subfolder",
        default=DEFAULT_SUBFOLDER_NAME,
        help=f"Subfolder name under parent (default: {DEFAULT_SUBFOLDER_NAME})",
    )
    parser.add_argument(
        "--login-only",
        action="store_true",
        help="Run the browser login flow and save token, then exit",
    )
    parser.add_argument(
        "--reauth",
        action="store_true",
        help="Force a fresh browser login (ignore saved token)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.login_only:
            get_gdrive_service(
                args.credentials,
                args.token,
                force_login=True,
            )
            print(f"Token saved to {args.token}")
            return 0

        upload_file_to_gdrive(
            args.local_file,
            credentials_file=args.credentials,
            token_file=args.token,
            parent_folder_id=args.parent_folder_id,
            subfolder_name=args.subfolder,
            force_login=args.reauth,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
