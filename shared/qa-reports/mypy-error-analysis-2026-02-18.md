# Mypy Type Checking Error Analysis Report

**Date:** 2026-02-18  
**Total Errors:** 847  
**Files Affected:** 80  
**Status:** ⚠️ REQUIRES ATTENTION

---

## Executive Summary

The codebase has **847 mypy type errors** across 80 files. The majority of errors fall into 5 major categories:

1. **Argument type mismatches** (150 errors) - Function calls with incompatible types
2. **Logging call argument issues** (134 errors) - Structured logging with extra parameters
3. **Missing function annotations** (115 errors) - Untyped function definitions
4. **Assignment type mismatches** (88 errors) - Primarily SQLAlchemy Column issues
5. **Missing generic type parameters** (72 errors) - dict, list, Callable without type args

---

## Top 5 Most Common Error Types

### 1. `arg-type` (150 errors) - Argument Type Mismatches

**Pattern:** Function calls passing SQLAlchemy `Column` types instead of actual values, or passing `Any` where specific types expected.

**Top Examples:**
- `Argument "request_id" to "create" of "ApiResponse" has incompatible type "Any | str"; expected "UUID"` (24 occurrences in main.py)
- `Argument "X" to "JobResponse" has incompatible type "Column[Y]"; expected "Y"` (common pattern)

**Root Cause:** ORM model instances are being passed directly to Pydantic/response constructors instead of accessing the attribute values.

**Fix Strategy:**
```python
# ❌ Wrong - passing Column types
return JobResponse(
    id=job.id,  # Column[UUID] instead of UUID
    status=job.status,  # Column[str] instead of str
)

# ✅ Correct - ensure proper typing or use model_dump
from sqlalchemy.orm import declarative_base
Base = declarative_base()

# Or use proper type annotations with mapped_column
```

---

### 2. `call-arg` (134 errors) - Unexpected Keyword Arguments

**Pattern:** All errors are related to structured logging using extra keyword arguments.

**Example:**
```python
# ❌ Wrong - standard logging.Logger doesn't accept extra kwargs
logger.info("Message", job_id="123", attempt=1)

# ✅ Correct - use extra dict or structured logger
logger.info("Message", extra={"job_id": "123", "attempt": 1})
# OR use a properly typed structured logger
```

**Affected Files:**
- `src/core/healing.py` - 15 errors
- `src/core/routing.py` - 12 errors  
- `src/core/retry.py` - 10 errors
- `src/core/learning.py` - 5 errors
- `src/core/dlq.py` - 7 errors

**Fix Strategy:** Either:
1. Switch to `extra=` dict parameter: `logger.info("msg", extra={"key": val})`
2. Add `**kwargs` to custom logger wrapper
3. Use properly typed structured logger from `structlog` or similar

---

### 3. `no-untyped-def` (115 errors) - Missing Function Annotations

**Pattern:** Functions missing return types, parameter types, or both.

**Top Files:**
- `src/db/models.py` - 8 errors (SQLAlchemy model methods)
- `src/auth/rbac.py` - 6 errors
- `src/core/graphrag/knowledge_graph.py` - 5 errors
- `src/plugins/sources/` - Multiple errors

**Fix Strategy:**
```python
# ❌ Wrong
async def process_data(data, options=None):
    pass

# ✅ Correct
async def process_data(
    data: dict[str, Any], 
    options: dict[str, Any] | None = None
) -> ProcessResult:
    pass
```

**Quick Fix:** Add `-> None` return types first (mypy suggests this), then gradually add parameter types.

---

### 4. `assignment` (88 errors) - Incompatible Type Assignments

**Sub-patterns:**

#### A. SQLAlchemy Column Assignments (40+ errors)
```python
# ❌ Wrong - assigning value to Column type
job.status = "completed"  # Column[str] = str
job.created_at = datetime.now()  # Column[datetime] = datetime

# ✅ Correct - use proper SQLAlchemy 2.0 mapped_column
from sqlalchemy.orm import Mapped, mapped_column

class Job(Base):
    status: Mapped[str] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(DateTime)
```

**Affected:**
- `src/db/repositories/job.py` - 30 errors
- `src/db/repositories/webhook.py` - 15 errors
- `src/db/repositories/pipeline.py` - 10 errors

#### B. Optional/Default None Issues
```python
# ❌ Wrong - PEP 484 prohibits implicit Optional
def func(user: User = None):  # [assignment] error

# ✅ Correct
def func(user: User | None = None):
```

**Fix:** Add `| None` to optional parameters or enable `implicit_optional = True` in mypy config.

---

### 5. `type-arg` (72 errors) - Missing Generic Type Parameters

**Pattern:** Using generic types without parameters: `dict`, `list`, `Callable`, `Queue`

**Common Examples:**
```python
# ❌ Wrong
def process(data: dict) -> list:  # [type-arg]
    callbacks: list = []  # [type-arg]
    
# ✅ Correct  
def process(data: dict[str, Any]) -> list[str]:
    callbacks: list[Callable[[], None]] = []
```

**Fix Strategy:** Run regex search-replace:
- `dict\b` → `dict[str, Any]` (or appropriate types)
- `list\b` → `list[Any]` (or appropriate types)
- `Callable\b` → `Callable[..., Any]`

---

## Files with Most Errors

| File | Error Count | Primary Issues |
|------|-------------|----------------|
| `src/main.py` | 149 | arg-type (JobResponse), type-arg (dict) |
| `src/db/models.py` | 43 | valid-type (Base), no-untyped-def, assignment |
| `src/core/engine.py` | 37 | arg-type, no-untyped-def |
| `src/core/queue.py` | 30 | call-arg (logging), arg-type |
| `src/core/webhook_delivery.py` | 25 | call-arg (logging) |
| `src/db/repositories/job.py` | 23 | assignment (Column types), return-value |
| `src/core/dlq.py` | 20 | call-arg (logging), assignment |
| `src/db/repositories/webhook.py` | 19 | assignment, return-value |
| `src/core/healing.py` | 18 | call-arg (logging) |
| `src/api/routes/detection.py` | 18 | arg-type, misc |

---

## Other Notable Error Patterns

### `union-attr` (48 errors) - None Safety Issues

**Pattern:** Accessing attributes on potentially None values.

```python
# ❌ Wrong
self.driver.session()  # driver could be None

# ✅ Correct
if self.driver is not None:
    self.driver.session()
# OR use assert
assert self.driver is not None
```

**Affected:**
- `src/plugins/destinations/neo4j.py` - 4 errors
- `src/observability/tracing.py` - 3 errors
- `src/services/hybrid_search_service.py` - 1 error

### `no-any-return` (38 errors) - Untyped Returns

Functions calling untyped code return `Any`, but function signature declares specific return type.

**Fix:** Add type annotations to called functions or use `cast()`:
```python
from typing import cast
return cast(str, untyped_function())
```

### `valid-type` (12 errors) - SQLAlchemy Base Issues

SQLAlchemy declarative base not properly recognized as a type.

**Fix:** Use `Mapped` and `mapped_column` from SQLAlchemy 2.0 style.

### `import-untyped` / `import-not-found` (18 errors)

Missing type stubs for:
- `yaml` → `pip install types-PyYAML`
- `jose` → `pip install types-python-jose`  
- `fitz` (PyMuPDF) - no stubs available, use `# type: ignore`
- `pytesseract` - no stubs, use `# type: ignore`
- `lxml` → `pip install lxml-stubs`
- `aiofiles` → `pip install types-aiofiles`
- `boto3`, `botocore`, `neo4j`, `ijson` - use `# type: ignore`

---

## Recommended Fix Priority

### 🔴 High Priority (Fix First)

1. **Fix logging call-arg errors (134)** - Quick win, affects 15+ files
   ```python
   # Replace: logger.info("msg", key=val)
   # With:    logger.info("msg", extra={"key": val})
   ```

2. **Fix type-arg errors (72)** - Mechanical fix with regex
   - `dict` → `dict[str, Any]`
   - `list` → `list[Any]`

3. **Fix main.py arg-type errors (149)** - Core API file
   - Access SQLAlchemy column values properly
   - Ensure UUID types match

### 🟡 Medium Priority

4. **Fix no-untyped-def errors (115)** - Add basic annotations
   - Start with `-> None` return types
   - Add parameter types gradually

5. **Fix SQLAlchemy assignment errors (88)** - Requires SQLAlchemy 2.0 migration or `# type: ignore`

### 🟢 Low Priority

6. **Install missing stubs** - Easy environmental fix
7. **Fix union-attr errors** - Add null checks
8. **Fix remaining misc errors**

---

## Quick Fix Commands

```bash
# 1. Install missing type stubs
pip install types-PyYAML types-python-jose lxml-stubs types-aiofiles

# 2. Add mypy config to pyproject.toml for easier management
# [tool.mypy]
# implicit_optional = true  # Reduces assignment errors
# ignore_missing_imports = true  # For untyped libraries

# 3. For SQLAlchemy issues, consider SQLAlchemy 2.0 style
# from sqlalchemy.orm import Mapped, mapped_column
```

---

## Risk Assessment

| Risk | Level | Impact |
|------|-------|--------|
| Type errors in main.py | High | API response serialization issues |
| SQLAlchemy type mismatches | Medium | Potential runtime errors on ORM operations |
| Logging errors | Low | Runtime works, just not type-safe |
| Missing generic types | Low | Reduces type safety |

---

## Conclusion

The type errors are primarily concentrated in:
1. **API serialization layer** (main.py) - Column types vs values
2. **Logging throughout codebase** - 134 call-arg errors
3. **SQLAlchemy ORM usage** - Assignment and type definition issues

**Recommended approach:**
1. Quick wins first: logging fixes + stub installations
2. Add mypy configuration to reduce strictness temporarily
3. Gradually fix SQLAlchemy patterns with proper 2.0 style
4. Set up mypy in CI with incremental enforcement

---

**QA Engineer:** qa-agent  
**Next Steps:** Address high priority fixes in sprints, add mypy to pre-commit hooks after error count < 100
