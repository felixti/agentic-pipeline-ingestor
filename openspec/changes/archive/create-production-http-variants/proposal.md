# Proposal: Create Production HTTP File Variants

## Problem Statement

The project has HTTP client files in the `http/` folder for testing the API endpoints using REST Client (VS Code extension). These files currently point to `http://localhost:8000/api/v1` which is only suitable for local development.

For production testing and validation, we need variants that point to the production deployment at `https://ag-dt-ppl-api.felixtek.cloud/`.

## Proposed Solution

Create production variants of all HTTP files in the `http/` folder:

1. Create a new `http/production/` directory
2. Copy and adapt all existing HTTP files to point to production URL
3. Update the `@baseUrl` variable to use HTTPS production endpoint
4. Add appropriate documentation headers for production context

## Success Criteria

- [ ] Production HTTP files created in `http/production/` directory
- [ ] All 5 HTTP file variants created:
  - `all-searches.http`
  - `hybrid.http`
  - `semantic.http`
  - `similar.http`
  - `text.http`
- [ ] Base URL points to `https://ag-dt-ppl-api.felixtek.cloud/api/v1`
- [ ] Files maintain same request structure and examples as local versions
- [ ] Production-specific warnings/documentation added

## Scope

**In Scope:**
- Creating production variants of HTTP files
- Updating base URL to production endpoint
- Adding production context documentation

**Out of Scope:**
- Changing API endpoints or request structures
- Adding new functionality
- Modifying local development HTTP files
