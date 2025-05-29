### **Step‑by‑step: create your Neon Free‑Tier project for EchoVault MCP**

\*(written for Conyers, GA; latency ≈ 10–15 ms to “Azure East US 2 (Virginia)”) \*

---

#### **1 · Create the Neon project**

| UI field | What to choose | Why |
| ----- | ----- | ----- |
| **Project name** | `echovault-dev` | any slug works; keep it short |
| **Postgres version** | `17` (default) | pgvector works fine on PG 17 |
| **Cloud provider** | `Azure` | free tier is identical; Azure East is geographically close |
| **Region** | `Azure East US 2 (Virginia)` | lowest ping from Conyers |

Click **Create project**.

---

#### **2 · Grab your connection string (DSN)**

1. Neon redirects to the **Dashboard → Branches → `main`**.

2. Click **“Connection Details”**.

3. Copy the **“Connection string”** that ends with `...@ep-<slug>.neon.tech/neondb`.  
    *Toggle “Include password”.*

**Paste this DSN** into `.bolt/.env` as  
 `NEON_DSN=postgresql://<user>:<password>@ep-…neon.tech/neondb`

*(You can create a new DB user later; the default owner is OK for dev.)*

---

#### **3 · Open SQL console & enable pgvector**

1. Neon left sidebar → **SQL Editor**.

2. Run:

CREATE EXTENSION IF NOT EXISTS "vector";

You should see `CREATE EXTENSION`.

*(This allows the `vector(1536)` column in our Alembic migration.)*

---

#### **4 · Set pooling / autoscaling defaults**

Neon Settings → **Compute & scaling**:

* Base compute: `0.25 CU` (free tier)

* Autoscale up to: `2 CU`

* Idle shutdown: **keep default 5 min** (cold‑start 200‑300 ms is fine)

*(No cost impact until you exceed 0.5 GB storage or 191.9 compute‑hrs.)*

---

#### **5 · Test locally (optional but recommended)**

\# install psql if you don't have it  
sudo apt-get install postgresql-client \-y   \# Debian/WSL  
\# macOS: brew install libpq && echo 'export PATH="/opt/homebrew/opt/libpq/bin:$PATH"' \>\> \~/.zshrc

psql \-d "postgresql://\<user\>:\<password\>@ep-...neon.tech/neondb"

neondb=\> \\dx   \-- should list 'vector'  
neondb=\> \\q

---

#### **6 · Create a service user (optional hardening)**

\-- in SQL Editor  
CREATE ROLE echovault\_rw LOGIN PASSWORD 'replace‑me';  
GRANT CONNECT ON DATABASE neondb TO echovault\_rw;  
GRANT USAGE ON SCHEMA public TO echovault\_rw;  
GRANT INSERT, SELECT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO echovault\_rw;  
ALTER DEFAULT PRIVILEGES IN SCHEMA public  
    GRANT INSERT, SELECT, UPDATE, DELETE ON TABLES TO echovault\_rw;

Update `NEON_DSN` to use `echovault_rw`.

---

#### **7 · Hook Neon DSN into Bolt.new**

1. **Commit** `.bolt/env.example` ➜ `.bolt/.env` locally (DON’T push secrets\!).

2. In Bolt UI → **Environment variables** tab, add `NEON_DSN` with the DSN you copied.

3. When Bolt enters Build Mode it will inject the DSN for tests & Alembic migrations.

---

#### **8 · Next steps in the roadmap**

* Milestone **M‑2**: Bolt creates `neon_client.py` and Alembic migration that uses the DSN.

* Milestone **M‑3+**: vector mirroring, blob off‑load, etc.

---

### **Quick recap ✅**

1. **Created** `echovault-dev` project in **Azure East US 2**.

2. **Enabled** `pgvector` extension.

3. **Copied** DSN → `.bolt/.env`.

4. **Optionally** created least‑privilege user.

5. Bolt is now unblocked for M‑2 tasks—no over‑engineering required.

You’re ready—switch Bolt to Discussion Mode and let it list tasks for Milestone M‑1\!

