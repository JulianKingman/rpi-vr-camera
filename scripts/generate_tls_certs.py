#!/usr/bin/env python3
"""Interactive helper for generating self-signed HTTPS certificates."""

from __future__ import annotations

import subprocess
import shutil
from pathlib import Path


DEFAULT_OUTPUT_DIR = Path("certs")
DEFAULT_CA_CN = "VR Dev CA"
DEFAULT_SERVER_CN = "rpi-vr-camera"
DEFAULT_SERVER_IP = "192.168.1.50"
DEFAULT_CA_DAYS = "3650"
DEFAULT_SERVER_DAYS = "825"


def prompt(message: str, default: str) -> str:
    value = input(f"{message} [{default}]: ").strip()
    return value or default


def confirm(message: str, default: bool = True) -> bool:
    suffix = "Y/n" if default else "y/N"
    choice = input(f"{message} ({suffix}): ").strip().lower()
    if not choice:
        return default
    return choice in {"y", "yes"}


def check_command_exists(command: str) -> None:
    if shutil.which(command) is None:
        raise SystemExit(f"Required command '{command}' not found. Install it and rerun.")


def run_command(args: list[str], cwd: Path | None = None) -> None:
    print(f"==> {' '.join(str(a) for a in args)}")
    subprocess.run(args, cwd=cwd, check=True)


def sanitize(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in value)


def main() -> None:
    try:
        check_command_exists("openssl")
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(str(exc)) from exc

    output_dir = Path(prompt("Output directory for certificates", str(DEFAULT_OUTPUT_DIR))).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    ca_common_name = prompt("Certificate Authority common name", DEFAULT_CA_CN)
    ca_days = prompt("CA validity days", DEFAULT_CA_DAYS)

    server_common_name = prompt("Server hostname (common name)", DEFAULT_SERVER_CN)
    server_slug = sanitize(server_common_name)
    server_days = prompt("Server certificate validity days", DEFAULT_SERVER_DAYS)

    include_mdns = confirm("Include <hostname>.local in SANs?", True)
    extra_dns = input("Additional DNS names (comma separated, optional): ").strip()
    extra_ips = input("IP addresses for SANs (comma separated, optional): ").strip()

    dns_entries = [server_common_name]
    if include_mdns:
        dns_entries.append(f"{server_common_name}.local")
    if extra_dns:
        dns_entries.extend([item.strip() for item in extra_dns.split(",") if item.strip()])
    dns_entries = list(dict.fromkeys(dns_entries))

    ip_entries: list[str] = []
    if extra_ips:
        ip_entries.extend([item.strip() for item in extra_ips.split(",") if item.strip()])
    elif confirm(f"Include default IP {DEFAULT_SERVER_IP}?", True):
        ip_entries.append(DEFAULT_SERVER_IP)
    ip_entries = list(dict.fromkeys(ip_entries))

    print("\nUsing the following Subject Alternative Names:")
    for dns in dns_entries:
        print(f"  DNS: {dns}")
    for ip in ip_entries:
        print(f"  IP:  {ip}")

    ca_key = output_dir / "dev-ca.key"
    ca_crt = output_dir / "dev-ca.crt"
    server_key = output_dir / f"{server_slug}-server.key"
    server_csr = output_dir / f"{server_slug}-server.csr"
    server_crt = output_dir / f"{server_slug}-server.crt"
    ext_file = output_dir / f"{server_slug}-server.ext"

    if ca_key.exists() or ca_crt.exists():
        if confirm("CA files already exist. Recreate them?", False):
            ca_key.unlink(missing_ok=True)
            ca_crt.unlink(missing_ok=True)
        else:
            print("Reusing existing CA files.")

    if not ca_key.exists() and not ca_crt.exists():
        run_command(
            [
                "openssl",
                "req",
                "-x509",
                "-nodes",
                "-newkey",
                "rsa:4096",
                "-keyout",
                str(ca_key),
                "-out",
                str(ca_crt),
                "-days",
                ca_days,
                "-subj",
                f"/CN={ca_common_name}",
            ]
        )
    else:
        print("Skipping CA generation.")

    if server_key.exists() or server_csr.exists() or server_crt.exists():
        if confirm("Server certificate files already exist. Recreate them?", False):
            server_key.unlink(missing_ok=True)
            server_csr.unlink(missing_ok=True)
            server_crt.unlink(missing_ok=True)
        else:
            print("Server files exist; skipping regeneration.")
            return

    run_command(
        [
            "openssl",
            "req",
            "-new",
            "-nodes",
            "-newkey",
            "rsa:2048",
            "-keyout",
            str(server_key),
            "-out",
            str(server_csr),
            "-subj",
            f"/CN={server_common_name}",
        ]
    )

    with ext_file.open("w", encoding="utf-8") as fh:
        fh.write("subjectAltName = @alt_names\n")
        fh.write("[alt_names]\n")
        for idx, dns in enumerate(dns_entries, start=1):
            fh.write(f"DNS.{idx} = {dns}\n")
        for idx, ip in enumerate(ip_entries, start=1):
            fh.write(f"IP.{idx} = {ip}\n")

    run_command(
        [
            "openssl",
            "x509",
            "-req",
            "-in",
            str(server_csr),
            "-CA",
            str(ca_crt),
            "-CAkey",
            str(ca_key),
            "-CAcreateserial",
            "-out",
            str(server_crt),
            "-days",
            server_days,
            "-sha256",
            "-extfile",
            str(ext_file),
        ]
    )

    print("\nTLS assets generated:")
    print(f"  CA certificate:     {ca_crt}")
    print(f"  CA private key:     {ca_key}")
    print(f"  Server certificate: {server_crt}")
    print(f"  Server key:         {server_key}")
    recommend_cmd = (
        'make stream-webrtc ARGS="--host 0.0.0.0 --port 8443 '
        f'--cert {server_crt} --key {server_key} --ca-cert {ca_crt}"'
    )
    print("\nInstall the CA certificate on your Quest, then launch:")
    print(f"  {recommend_cmd}")


if __name__ == "__main__":
    main()
