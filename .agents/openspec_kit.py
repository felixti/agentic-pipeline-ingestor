#!/usr/bin/env python3
"""
OpenSpec Kit - Lightweight state manager for OpenSpec workflow.
Replaces the openspec CLI for Kimi Code compatibility.
"""

import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

# Default schema definitions (embedded for portability)
SCHEMAS = {
    "spec-driven": {
        "name": "spec-driven",
        "artifacts": [
            {
                "id": "proposal",
                "name": "Proposal",
                "outputPath": "proposal.md",
                "dependsOn": [],
                "template": """# Proposal: {change_name}

## Why
[Describe the problem or opportunity]

## What Changes
[High-level description of changes]

## Capabilities
[List capabilities that need specs]
- [ ] Capability 1
- [ ] Capability 2

## Impact
[Impact assessment]
"""
            },
            {
                "id": "spec",
                "name": "Spec",
                "outputPath": "specs/{capability}/spec.md",
                "dependsOn": ["proposal"],
                "multiple": True,  # One per capability
                "template": """# Spec: {capability}

## Overview
[Capability description]

### Requirements

#### Requirement 1
**Given** [context]
**When** [action]
**Then** [expected result]

### Scenarios

#### Scenario 1
[Detailed scenario description]
"""
            },
            {
                "id": "design",
                "name": "Design",
                "outputPath": "design.md",
                "dependsOn": ["spec"],
                "template": """# Design: {change_name}

## Architecture
[Architecture decisions]

## Approach
[Implementation approach]

## Decisions
- Decision 1: [rationale]
"""
            },
            {
                "id": "tasks",
                "name": "Tasks",
                "outputPath": "tasks.md",
                "dependsOn": ["design"],
                "template": """# Tasks: {change_name}

## Implementation Tasks

- [ ] Task 1
- [ ] Task 2
- [ ] Task 3
"""
            }
        ],
        "applyRequires": ["tasks"]  # Artifacts needed before implementation
    }
}


def get_openspec_dir() -> Path:
    """Get the OpenSpec root directory."""
    return Path("openspec")


def get_changes_dir() -> Path:
    """Get the changes directory."""
    return get_openspec_dir() / "changes"


def get_archive_dir() -> Path:
    """Get the archive directory."""
    return get_changes_dir() / "archive"


def validate_name(name: str) -> bool:
    """Validate change name (kebab-case)."""
    return bool(re.match(r'^[a-z0-9]+(-[a-z0-9]+)*$', name))


def create_change(name: str, schema: str = "spec-driven") -> dict:
    """Create a new change directory.
    
    Equivalent to: openspec new change <name>
    """
    if not validate_name(name):
        raise ValueError(f"Invalid change name '{name}'. Use kebab-case (e.g., 'my-change')")
    
    changes_dir = get_changes_dir()
    change_dir = changes_dir / name
    
    if change_dir.exists():
        raise FileExistsError(f"Change '{name}' already exists at {change_dir}")
    
    if schema not in SCHEMAS:
        raise ValueError(f"Unknown schema '{schema}'. Available: {list(SCHEMAS.keys())}")
    
    # Create directory structure
    change_dir.mkdir(parents=True)
    specs_dir = change_dir / "specs"
    specs_dir.mkdir()
    
    # Create metadata file
    metadata = {
        "name": name,
        "schema": schema,
        "created": datetime.now().isoformat(),
        "modified": datetime.now().isoformat()
    }
    
    metadata_path = change_dir / ".openspec.yaml"
    with open(metadata_path, 'w') as f:
        import yaml
        yaml.dump(metadata, f, default_flow_style=False)
    
    return {
        "name": name,
        "path": str(change_dir),
        "schema": schema,
        "status": "created"
    }


def get_change_metadata(change_path: Path) -> dict:
    """Read change metadata."""
    metadata_path = change_path / ".openspec.yaml"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Not a valid change: {change_path}")
    
    import yaml
    with open(metadata_path) as f:
        return yaml.safe_load(f)


def artifact_exists(change_path: Path, artifact: dict, capability: str = None) -> bool:
    """Check if an artifact file exists."""
    output_path = artifact["outputPath"]
    if capability:
        output_path = output_path.format(capability=capability)
    
    artifact_path = change_path / output_path
    return artifact_path.exists()


def get_status(name: str) -> dict:
    """Get change status.
    
    Equivalent to: openspec status --change <name> --json
    """
    changes_dir = get_changes_dir()
    change_path = changes_dir / name
    
    if not change_path.exists():
        raise FileNotFoundError(f"Change '{name}' not found")
    
    metadata = get_change_metadata(change_path)
    schema_name = metadata.get("schema", "spec-driven")
    schema = SCHEMAS.get(schema_name, SCHEMAS["spec-driven"])
    
    artifacts_status = []
    capabilities = []
    
    # Check for capabilities from proposal
    proposal_path = change_path / "proposal.md"
    if proposal_path.exists():
        content = proposal_path.read_text()
        # Extract capabilities from ## Capabilities section
        caps_match = re.search(r'## Capabilities\s*\n(.*?)(?:\n## |\Z)', content, re.DOTALL)
        if caps_match:
            caps_section = caps_match.group(1)
            capabilities = re.findall(r'- \[.]\s*([^(\n]+)', caps_section)
            capabilities = [c.strip().lower().replace(' ', '-') for c in capabilities]
    
    for artifact in schema["artifacts"]:
        if artifact.get("multiple") and capabilities:
            # Check each capability instance
            for cap in capabilities:
                exists = artifact_exists(change_path, artifact, cap)
                artifacts_status.append({
                    "id": f"{artifact['id']}/{cap}",
                    "name": f"{artifact['name']} ({cap})",
                    "status": "done" if exists else "ready" if all(
                        any(a["id"] == dep and a.get("status") == "done" for a in artifacts_status)
                        for dep in artifact.get("dependsOn", [])
                    ) else "blocked",
                    "outputPath": artifact["outputPath"].format(capability=cap)
                })
        else:
            exists = artifact_exists(change_path, artifact)
            deps_satisfied = all(
                any(a["id"] == dep and a.get("status") == "done" for a in artifacts_status)
                for dep in artifact.get("dependsOn", [])
            ) if artifact.get("dependsOn") else True
            
            status = "done" if exists else ("ready" if deps_satisfied else "blocked")
            
            artifacts_status.append({
                "id": artifact["id"],
                "name": artifact["name"],
                "status": status,
                "outputPath": artifact["outputPath"]
            })
    
    # Check if all apply-requires artifacts are done
    apply_requires = schema.get("applyRequires", [])
    is_apply_ready = all(
        any(a["id"] == req and a["status"] == "done" for a in artifacts_status)
        for req in apply_requires
    )
    
    is_complete = all(a["status"] == "done" for a in artifacts_status)
    
    return {
        "name": name,
        "schemaName": schema_name,
        "path": str(change_path),
        "artifacts": artifacts_status,
        "isComplete": is_complete,
        "isApplyReady": is_apply_ready,
        "applyRequires": apply_requires
    }


def get_instructions(name: str, artifact_id: str) -> dict:
    """Get instructions for creating an artifact.
    
    Equivalent to: openspec instructions <artifact> --change <name> --json
    """
    changes_dir = get_changes_dir()
    change_path = changes_dir / name
    
    if not change_path.exists():
        raise FileNotFoundError(f"Change '{name}' not found")
    
    metadata = get_change_metadata(change_path)
    schema_name = metadata.get("schema", "spec-driven")
    schema = SCHEMAS.get(schema_name, SCHEMAS["spec-driven"])
    
    # Find artifact definition
    artifact_def = None
    base_artifact_id = artifact_id
    capability = None
    
    if "/" in artifact_id:
        base_artifact_id, capability = artifact_id.split("/", 1)
    
    for art in schema["artifacts"]:
        if art["id"] == base_artifact_id:
            artifact_def = art
            break
    
    if not artifact_def:
        raise ValueError(f"Unknown artifact: {artifact_id}")
    
    # Get context from dependencies
    context_files = []
    for dep in artifact_def.get("dependsOn", []):
        dep_path = change_path / f"{dep}.md"
        if dep_path.exists():
            context_files.append(str(dep_path))
        # Also check for specs directory
        dep_dir = change_path / dep
        if dep_dir.exists():
            for spec_file in dep_dir.rglob("*.md"):
                context_files.append(str(spec_file))
    
    # Read context content
    context_content = ""
    for ctx_file in context_files:
        ctx_path = Path(ctx_file)
        if ctx_path.exists():
            context_content += f"\n\n--- From {ctx_path.name} ---\n\n"
            context_content += ctx_path.read_text()
    
    # Format template
    template = artifact_def["template"]
    if capability:
        template = template.format(capability=capability, change_name=name)
    else:
        template = template.format(change_name=name)
    
    output_path = artifact_def["outputPath"]
    if capability:
        output_path = output_path.format(capability=capability)
    
    return {
        "artifactId": artifact_id,
        "name": artifact_def["name"],
        "context": context_content,
        "template": template,
        "instruction": f"Create the {artifact_def['name']} artifact for {name}",
        "outputPath": str(change_path / output_path),
        "dependencies": context_files,
        "changePath": str(change_path)
    }


def list_changes() -> list[dict]:
    """List all changes.
    
    Equivalent to: openspec list --json
    """
    changes_dir = get_changes_dir()
    
    if not changes_dir.exists():
        return []
    
    changes = []
    for change_dir in changes_dir.iterdir():
        if change_dir.is_dir() and not change_dir.name == "archive":
            try:
                metadata = get_change_metadata(change_dir)
                status = get_status(change_dir.name)
                
                done_count = sum(1 for a in status["artifacts"] if a["status"] == "done")
                total_count = len(status["artifacts"])
                
                changes.append({
                    "name": change_dir.name,
                    "schema": metadata.get("schema", "spec-driven"),
                    "status": f"{done_count}/{total_count} artifacts",
                    "lastModified": metadata.get("modified", metadata.get("created", "")),
                    "isComplete": status["isComplete"]
                })
            except Exception:
                # Skip invalid changes
                pass
    
    # Sort by last modified (most recent first)
    changes.sort(key=lambda x: x["lastModified"], reverse=True)
    return changes


def archive_change(name: str) -> dict:
    """Archive a change.
    
    Equivalent to: mv openspec/changes/<name> openspec/changes/archive/YYYY-MM-DD-<name>
    """
    changes_dir = get_changes_dir()
    archive_dir = get_archive_dir()
    change_path = changes_dir / name
    
    if not change_path.exists():
        raise FileNotFoundError(f"Change '{name}' not found")
    
    archive_dir.mkdir(parents=True, exist_ok=True)
    
    date_prefix = datetime.now().strftime("%Y-%m-%d")
    archive_name = f"{date_prefix}-{name}"
    archive_path = archive_dir / archive_name
    
    if archive_path.exists():
        raise FileExistsError(f"Archive '{archive_name}' already exists")
    
    shutil.move(str(change_path), str(archive_path))
    
    return {
        "name": name,
        "archivedTo": str(archive_path),
        "archiveName": archive_name
    }


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python openspec_kit.py <command> [args...]")
        print("Commands: create, status, instructions, list, archive")
        sys.exit(1)
    
    command = sys.argv[1]
    
    try:
        if command == "create" and len(sys.argv) >= 3:
            schema = sys.argv[3] if len(sys.argv) > 3 else "spec-driven"
            result = create_change(sys.argv[2], schema)
            print(json.dumps(result, indent=2))
        
        elif command == "status" and len(sys.argv) >= 3:
            result = get_status(sys.argv[2])
            print(json.dumps(result, indent=2))
        
        elif command == "instructions" and len(sys.argv) >= 4:
            result = get_instructions(sys.argv[3], sys.argv[2])
            print(json.dumps(result, indent=2))
        
        elif command == "list":
            result = list_changes()
            print(json.dumps(result, indent=2))
        
        elif command == "archive" and len(sys.argv) >= 3:
            result = archive_change(sys.argv[2])
            print(json.dumps(result, indent=2))
        
        else:
            print(f"Unknown command or missing arguments: {command}")
            sys.exit(1)
    
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)
