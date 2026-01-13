# Bottleneck Analysis: Loading Reasoning Examples in app.py

## Summary
The `load_sample_cots_from_wandb()` function has several critical performance bottlenecks that cause slow loading times.

## Main Bottlenecks

### 1. **`run.files()` - The Biggest Bottleneck** (Line 620)
```python
files = list(run.files())
```

**Problem:**
- Fetches metadata for **ALL files** in each run (can be thousands of files)
- Makes API calls to W&B for each file's metadata
- No filtering at the API level - all files are fetched first, then filtered
- Executed for **every run** in **every project**

**Impact:** If a run has 1000 files, this makes 1000+ API requests just to get file metadata, even though you only need files matching `sample_cots`.

**Time complexity:** O(projects × runs × total_files_per_run)

---

### 2. **`run.history()` Called in Inner Loop** (Line 703)
```python
history = run.history(keys=["step"])
```

**Problem:**
- Called inside the file processing loop (line 630-721)
- Executed for each file that doesn't have a step in the filename
- Makes an additional API call for each such file

**Impact:** Multiple redundant API calls for the same run's history.

---

### 3. **`run.scan_history()` - Slow Fallback** (Line 730)
```python
for row in run.scan_history(keys=["step", "eval/sample_cots"]):
```

**Problem:**
- Scans through **ALL history rows** of the run
- Executed as a fallback when no files are found
- Can be very slow for long-running experiments

**Impact:** For runs with thousands of steps, this scans all of them even if sample_cots was only logged a few times.

---

### 4. **Nested Loops with API Calls**
```python
for project in projects:              # Outer loop
    for run_name, run_info in runs_by_name.items():  # Middle loop
        files = list(run.files())     # API call per run
        for file in table_files:      # Inner loop
            file_path = file.download(replace=True)  # Download per file
```

**Problem:**
- Triple nested loops with API calls at multiple levels
- No parallelization
- Sequential processing

**Impact:** Time multiplies across projects, runs, and files.

---

### 5. **File Downloads in Loop** (Line 633)
```python
file_path = file.download(replace=True)
```

**Problem:**
- Downloads files sequentially
- No caching between runs
- `replace=True` forces re-download even if file exists

---

## Performance Estimates

For a typical scenario:
- 3 projects
- 5 runs per project
- 1000 files per run (typical for training runs)
- 10 sample_cots files per run

**Current approach:**
- `run.files()`: 3 × 5 × 1000 = 15,000 file metadata fetches
- File downloads: 3 × 5 × 10 = 150 downloads
- Total: ~15,150 API operations

**Optimized approach (see recommendations):**
- Filtered file search: 3 × 5 × 10 = 150 targeted fetches
- File downloads: 150 downloads
- Total: ~300 API operations

**Speedup: ~50x faster**

---

## Recommendations

### High Priority

1. **Use W&B API filtering for files** (if available)
   - Check if `run.files()` supports pattern matching
   - Filter by filename pattern before fetching metadata

2. **Cache run.history() results**
   - Fetch once per run, reuse for all files in that run
   - Move outside the file processing loop

3. **Use W&B Tables API directly**
   - W&B tables may have a direct API endpoint
   - Check `wandb.Api().artifact()` or table-specific endpoints

4. **Parallelize file downloads**
   - Use `concurrent.futures.ThreadPoolExecutor` for file downloads
   - Batch operations

5. **Skip scan_history if files found**
   - Currently tries both methods sequentially
   - Should exit early if Method 1 succeeds

### Medium Priority

6. **Increase cache TTL**
   - Current: `@st.cache_data(ttl=60)` (60 seconds)
   - Consider: `ttl=3600` (1 hour) since data doesn't change often

7. **Lazy loading**
   - Only load examples when the tab is actually viewed
   - Don't load all projects at once

8. **Batch file operations**
   - Group files by run and process in batches

### Code-Level Optimizations

9. **Early exit conditions**
   - Stop processing a run if enough examples are found
   - Skip runs that are unlikely to have sample_cots

10. **Use file size as filter**
    - Skip very large files (unlikely to be sample_cots tables)
    - Sample_cots files are typically small JSON files

---

## Quick Wins

1. **Move `run.history()` outside the file loop** - Easy fix, immediate improvement
2. **Add early exit if `tables_found > 0`** - Prevents unnecessary scan_history
3. **Increase cache TTL to 3600 seconds** - Reduces redundant API calls
4. **Use file filtering pattern if W&B API supports it**

---

## Next Steps

1. Check W&B API documentation for file filtering options
2. Test if `run.file(pattern="**/sample_cots*.json")` or similar works
3. Implement caching of `run.history()` results
4. Add parallelization for file downloads
5. Profile the function to measure actual improvements

