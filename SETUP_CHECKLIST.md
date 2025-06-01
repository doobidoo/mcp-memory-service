# EchoVault Setup Checklist

## 🔧 Immediate Setup Tasks

### 0. Virtual Environment Setup
- [ ] Check Python version (3.11+ recommended)
  ```
  python --version
  ```
- [ ] Create virtual environment
  ```
  python -m venv .venv
  # OR for conda users
  conda create -n echovault python=3.11
  ```
- [ ] Activate virtual environment
  ```
  # Windows
  .venv\Scripts\activate
  # Linux/Mac
  source .venv/bin/activate
  # Conda
  conda activate echovault
  ```
- [ ] Verify activation
  ```
  # Should show .venv path
  pip --version
  ```

### 1. Environment Configuration
- [ ] Navigate to project directory
  ```
  cd "C:\Users\bjorn\Downloads\project-bolt-github-6wubd9uq\project"
  ```
- [ ] Create .env file from template
  ```
  copy .env.example .env
  ```
- [ ] Edit .env and add your credentials:
  - [ ] `NEON_DSN=<YOUR_NEON_DATABASE_URL>`
  - [ ] `QDRANT_URL=<YOUR_QDRANT_CLOUD_URL>`
  - [ ] `QDRANT_API_KEY=<YOUR_QDRANT_API_KEY>`
  - [ ] `R2_ENDPOINT=<YOUR_R2_ENDPOINT_URL>`
  - [ ] `R2_ACCESS_KEY_ID=<YOUR_R2_ACCESS_KEY_ID>`
  - [ ] `R2_SECRET_ACCESS_KEY=<YOUR_R2_SECRET_ACCESS_KEY>`
  - [ ] `R2_BUCKET=<YOUR_R2_BUCKET_NAME>`
  - [ ] `JWT_SECRET=<YOUR_SECURE_JWT_SECRET>`

### 2. Python Environment Setup
- [ ] Install overlay dependencies
  ```
  pip install -r requirements_overlay.txt
  ```
- [ ] Install base dependencies if needed
  ```
  pip install -r requirements.txt
  ```

### 3. Test Connectivity
- [ ] Run connectivity test
  ```
  python src\mcp_memory_service\test_connectivity.py
  ```
- [ ] Verify all services connect:
  - [ ] Neon PostgreSQL ✓
  - [ ] Qdrant Cloud ✓
  - [ ] Cloudflare R2 ✓

### 4. Database Setup
- [ ] Install Alembic if not installed
  ```
  pip install alembic
  ```
- [ ] Run database migrations
  ```
  alembic upgrade head
  ```

### 5. Test EchoVault
- [ ] Run in standard mode (ChromaDB)
  ```
  python memory_wrapper.py
  ```
- [ ] Run with EchoVault enabled
  ```
  python memory_wrapper.py --use-echovault
  ```

### 6. Run Tests
- [ ] Install pytest if needed
  ```
  pip install pytest pytest-asyncio
  ```
- [ ] Run test suite
  ```
  pytest tests/test_echovault_integration.py -v
  ```

## 🐳 Docker Setup (Optional)
- [ ] Build Docker image
  ```
  docker-compose build
  ```
- [ ] Start all services
  ```
  docker-compose up
  ```
- [ ] Access services:
  - [ ] API: http://localhost:8000
  - [ ] Jaeger: http://localhost:16686
  - [ ] Prometheus: http://localhost:9090
  - [ ] Grafana: http://localhost:3000

## 🔌 Claude Desktop Integration
- [ ] Update Claude Desktop config
- [ ] Point to your local server
- [ ] Add `--use-echovault` flag
- [ ] Test memory operations

## 📝 Notes
- If any connectivity test fails, check your .env credentials
- Neon might have a cold start delay (200-300ms)
- R2 bucket must exist (echovault-events)
- Qdrant free tier is single-node (no HA)

## 🎉 Success Indicators
- All connectivity tests pass
- Database migrations complete
- Can store and retrieve memories
- Observability metrics visible
- Large content stored in R2

Good luck! 🚀
