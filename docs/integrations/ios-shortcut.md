# iOS Shortcut: save to memory from the share sheet

A four-action Apple Shortcut that sends selected text, a URL, or a typed note from an
iPhone or iPad to `POST /api/memories`. No app, no extra service: the shortcut talks to
the HTTP API directly.

This page describes how to build the shortcut instead of shipping a `.shortcut` file.
A signed shortcut is a binary plist that cannot be reviewed before import, and the one
you build carries your API key, so it should never be shared anyway.

## Before you start: the phone has to reach the server

The HTTP server binds to `127.0.0.1` by default (`MCP_HTTP_HOST`), so a phone cannot
reach it out of the box. iOS also refuses self-signed and mkcert certificates unless
their CA is installed and trusted on the device. Pick one of these:

| Option | Reachable from | Certificate | Server change |
|--------|----------------|-------------|---------------|
| **Tailscale `serve`** (recommended) | your tailnet, at home and on the road | real Let's Encrypt cert on `<host>.<tailnet>.ts.net` | none |
| **LAN + mkcert CA on the phone** | home network only | mkcert, after installing its root CA as a profile and enabling full trust | `MCP_HTTP_HOST=0.0.0.0` |

### Tailscale `serve`

1. In the Tailscale admin console, enable **DNS > HTTPS Certificates** for the tailnet.
2. On the machine running the memory service:

   ```bash
   tailscale serve --bg https+insecure://127.0.0.1:8000
   ```

   `https+insecure` is needed when the service itself runs HTTPS with a local
   certificate; use `http://127.0.0.1:8000` if it runs plain HTTP. The service stays
   bound to localhost, and only devices in your tailnet can reach the forwarded port.
   `tailscale serve reset` removes it.
3. The base URL for the shortcut is `https://<host>.<tailnet>.ts.net`
   (`tailscale status --json` shows the full name under `Self.DNSName`).

### LAN with the mkcert CA

1. Create a certificate that lists the address the phone will use:

   ```bash
   mkcert -cert-file certs/lan.pem -key-file certs/lan-key.pem <lan-ip> <hostname>.local
   ```

2. Point the server at it and open it to the network, then restart:

   ```bash
   MCP_HTTP_HOST=0.0.0.0
   MCP_HTTPS_ENABLED=true
   MCP_SSL_CERT_FILE=/path/to/certs/lan.pem
   MCP_SSL_KEY_FILE=/path/to/certs/lan-key.pem
   ```

   The server is now reachable by anyone on the network, so the API key is the only
   thing protecting it.
3. Copy `$(mkcert -CAROOT)/rootCA.pem` to the phone, install it as a profile
   (Settings > General > VPN & Device Management), then enable it under
   Settings > General > About > Certificate Trust Settings.
4. The base URL is `https://<lan-ip>:8000`.

## Test the endpoint first

From a machine that can reach the same URL:

```bash
curl -X POST "<base-url>/api/memories" \
  -H "Authorization: Bearer $MCP_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"content": "Shortcut test", "tags": ["ios-shortcut"], "metadata": {"source": "ios-shortcut"}}'
```

A response with `"success": true` and a `content_hash` means the shortcut will work too.
`401` means the key is wrong. A TLS error means the certificate is not trusted yet.

## Build the shortcut

In the Shortcuts app, create a new shortcut named **Save to Memory** and add these
actions. Names follow the English iOS 18 UI.

1. **Receive** `Text` and `URLs` **input from** `Share Sheet`.
   Set **If there's no input** to **Ask For** `Text`, so the shortcut also works when
   started on its own (home screen, Siri, Action button).
2. **Get Contents of URL**
   - URL: `<base-url>/api/memories`
   - Method: `POST`
   - Headers:
     - `Authorization`: `Bearer <your MCP_API_KEY>`
     - `Content-Type`: `application/json`
   - Request Body: `JSON`, with these fields:
     - `content` (Text): the **Shortcut Input** variable
     - `tags` (Array): one Text item `ios-shortcut`; add more if you like
     - `metadata` (Dictionary): `source` (Text) = `ios-shortcut`
3. **Get Dictionary Value**: get `Value` for `message` in **Contents of URL**.
4. **Show Notification** with the **Dictionary Value** from step 3. The server answers
   `Memory stored successfully`, or says why it did not store, for example
   `Duplicate content detected (exact match)`.

Then open the shortcut's details and enable **Show in Share Sheet**.

## Limits

- The API key sits in plain text inside the shortcut. Do not share or export the
  shortcut after adding it, and rotate the key if you did.
- Photos and files are not handled. Only text and URLs are sent; a shared URL is stored
  as the URL itself, not the page content.
- Sharing the same text twice stores it once; the second run reports the duplicate.
