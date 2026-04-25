# How to Copy Files from Remote Server to Local Machine

## Options

### Option 1: scp (Recommended for simple copies)

```bash
# From YOUR LOCAL machine (not SSH):
scp -r user@remote_host:/home/phoenix/robin/* /path/to/local/folder/

# Example (if your SSH host is example.com):
scp -r phoenix@example.com:/home/phoenix/robin/* ./
```

### Option 2: rsync (Better for large files)

```bash
# From YOUR LOCAL machine:
rsync -avz user@remote_host:/home/phoenix/robin/ /path/to/local/folder/
```

### Option 3: sftp (Interactive)

```bash
# From YOUR LOCAL machine:
sftp user@remote_host

# Then in sftp:
sftp> get -r /home/phoenix/robin/* ./
sftp> quit
```

---

## What Files to Copy

All the files in `/home/phoenix/robin/`:

| File | Description |
|------|-------------|
| `README.md` | Main explanation |
| `TASK_A_SOLUTION.md` | Task A detailed solution |
| `TASK_A_OPINION.md` | Task A opinion |
| `TASK_B_SOLUTION.md` | Task B detailed solution |
| `TASK_B_OPINION.md` | Task B opinion |
| `TASK_C_SOLUTION.md` | Task C detailed solution |
| `TASK_C_OPINION.md` | Task C opinion |
| `counterfactual.json` | Counterfactual data |
| `persistence_report.pdf` | PDF report |
| `audit_scripts/` | Python scripts folder |
| `persistence_data.json` | Analysis data |

---

## Quick Copy All Files

Run this from **your local machine**:

```bash
# Create local folder
mkdir -p ~/robin_audit
cd ~/robin_audit

# Copy all files
scp -r phoenix@YOUR_SERVER:/home/phoenix/robin/* ./

# Alternatively, if using a specific SSH key:
scp -i ~/.ssh/key.pem -r user@host:/path/to/files ./
```

---

## If You Don't Know Your SSH Host

You're on SSH, so you already connected somehow. Check:

```bash
# Show your current SSH connection:
echo $SSH_CONNECTION
# or
who -m
```

---

## After Copying

Verify files:

```bash
ls -la
ls audit_scripts/
```

All done!