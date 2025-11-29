# Pre-Push Checklist for CrashSeverityNet

Before pushing to GitHub, please verify the following:

## 🔒 Data & Secrets
- [ ] No raw datasets (e.g., `US_Accidents_March23.csv`, `*.csv`) are tracked by git
- [ ] No model weights (`.pt`, `.pth`, `.ckpt`, `.joblib`) are tracked
- [ ] No experimental results (`results/`, `logs/`) are tracked
- [ ] No API keys, tokens, or `.env` files are tracked
- [ ] No personal email addresses, school accounts, or credentials in code/docs

## 📂 File Structure
- [ ] `.gitignore` is comprehensive and up-to-date
- [ ] `README.md` clearly states dataset must be downloaded separately from Kaggle
- [ ] `LICENSE` file exists (MIT License)
- [ ] `docs/thesis_summary.md` includes public disclaimer header

## 🔍 Code Quality
- [ ] No hardcoded absolute paths (e.g., `C:\Users\...`) in code
- [ ] Data paths use CLI arguments (e.g., `--data_path`) or relative paths
- [ ] All personal information anonymized (use `User A`, `University X`, etc.)
- [ ] No direct copy-paste from external papers without proper attribution

## 📝 Documentation
- [ ] `README.md` is complete with installation, usage, and citation instructions
- [ ] `docs/experiment_guide.md` provides clear reproduction steps
- [ ] `docs/literature_gap_analysis.md` properly cites all sources
- [ ] `docs/thesis_summary.md` clarifies that results are hypothetical/preliminary

## 🧪 Final Verification
Run these commands to double-check:

```bash
# Check what will be committed
git status

# Search for potential secrets (adjust patterns as needed)
grep -r "api_key" .
grep -r "@gmail.com" .
grep -r "C:\\Users" .

# Verify .gitignore is working
git ls-files | grep -E "\.(csv|pt|pth|env)$"  # Should return nothing
```

## ✅ Ready to Push

Once all items are checked, you can safely push to GitHub:

```bash
git add .
git commit -m "Initial public release of CrashSeverityNet"
git push origin main
```

---

**Note**: If you find any issues after pushing, you can force-remove files from history using:
```bash
# Remove a file from all commits
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch path/to/file" \
  --prune-empty --tag-name-filter cat -- --all
```

But it's best to check beforehand! 🚀
