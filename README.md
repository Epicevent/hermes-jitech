# JI TECH Hermes

This repository is the JI TECH Hermes product source repository.

It is not the server operations repository. Host install, NAS mount policy, customer slots, image catalog, rollout, drift checks, and the admin console are handled by [`Epicevent/openclaw-nas-agent-baseline`](https://github.com/Epicevent/openclaw-nas-agent-baseline).

## Responsibility

This repository owns:

- Hermes UI and branding changes
- Hermes provider/model UX
- Hermes NAS workspace behavior
- customer-facing Hermes behavior
- the source commit used to build a Hermes product image

This repository does not own:

- customer slot assignment
- canary slot selection
- production lane rollout
- NAS credentials
- Gemini/API keys
- gateway tokens
- `/srv/openclaw-ops`
- Apache vhost files
- server `.env` files

Secrets and customer data must not be committed here.

## Account And Environment Boundaries

The source repository, server development checkout, dev preview slot, and customer slots are different layers.

| Layer | Example | Purpose | Source of truth |
| --- | --- | --- | --- |
| Product source | `Epicevent/hermes-jitech` | Hermes code that JI TECH changes | This repository |
| Server development checkout | `/home/openclawdev/src/hermes-jitech` | Server-side working copy used by the developer account | This repository after push/pull |
| Dev preview slot | configured in operations | Shows the development build through Apache | Operations repo and `/srv/openclaw-ops` |
| Customer slot | configured in operations | Runs a published image digest | Operations repo and `/srv/openclaw-ops` |

The developer account and the dev preview slot are not the same thing.

```text
developer account:
  owns and edits the product source checkout
  may run build and image release work

dev preview slot:
  managed slot used to inspect the development build in a browser
  may use source mode when operations policy allows it

customer slot:
  managed slot used by a real tester or customer
  must run only a published registry image digest
```

This repository must not decide which customer slot is used for canary. That decision belongs to operations state.

## Development Loop

Development happens in the product source checkout.

```bash
cd /home/openclawdev/src/hermes-jitech
git status
```

On Windows, use the PowerShell installer at `scripts/install.ps1`.

The intended loop is:

```text
edit source
  -> build/update dev output
  -> inspect through the configured dev preview URL
  -> commit source
  -> push source
  -> publish product image from that commit
  -> operations wrapper image
  -> operations-selected canary slot
  -> rollout only after canary passes
```

Customer slots do not use source mode.

## Product Image Release

This repository publishes the Hermes product image to GHCR.

```text
ghcr.io/epicevent/hermes-jitech:<tag>
```

The workflow publishes `main` and `latest` on `main` branch updates. GitHub releases publish the release tag.

Use the architecture digest that matches the target server. The image tag is a human-readable name; digest is the deployment identity.

## Operations Wrapper Image

Customer slots do not run this product image directly. They run the Hermes NAS Agent wrapper image built by the operations repository.

```text
hermes-jitech source commit
  -> ghcr.io/epicevent/hermes-jitech:<tag>
  -> openclaw-nas-agent-baseline wrapper workflow
  -> ghcr.io/epicevent/openclaw-nas-agent:<release>
  -> server image catalog
  -> operations-selected canary slot
  -> Hermes lane rollout
```

## Server Registration

Server registration and rollout are operations work. Use the operations repository and `svcops-control.sh`.

This repository only produces the product source and product image.

## OpenClaw And Hermes Separation

OpenClaw and Hermes are separate product lanes.

```text
OpenClaw source:
  Epicevent/openclaw-jitech

Hermes source:
  Epicevent/hermes-jitech
```

Do not apply an OpenClaw image to the Hermes lane. Do not apply a Hermes image to the OpenClaw lane.

## Local Checks

```bash
git status
git log --oneline -5
```

Deployment state is verified in the operations repository and server image catalog, not by this repository alone.
