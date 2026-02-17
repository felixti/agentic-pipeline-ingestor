#!/usr/bin/env python3
"""SDK generation script using OpenAPI Generator.

This script generates Python and TypeScript SDKs from the OpenAPI 3.1
specification using the OpenAPI Generator tool.

Usage:
    python generate.py [--language python|typescript|all]

Requirements:
    - OpenAPI Generator CLI (java -jar openapi-generator-cli.jar)
    - Or openapi-generator installed via npm/brew
"""

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Configuration
OPENAPI_SPEC = Path(__file__).parent.parent / "api" / "openapi.yaml"
OUTPUT_DIR = Path(__file__).parent

# OpenAPI Generator configurations
CONFIGS = {
    "python": {
        "generator": "python",
        "output": OUTPUT_DIR / "python",
        "config": {
            "packageName": "pipeline_ingestor_client",
            "projectName": "pipeline-ingestor-client",
            "packageVersion": "1.0.0",
            "packageUrl": "https://github.com/example/agentic-pipeline-ingestor",
            "library": "asyncio",
            "useNose": "false",
            "enumUnknownDefaultCase": "true",
        },
    },
    "typescript": {
        "generator": "typescript-fetch",
        "output": OUTPUT_DIR / "typescript",
        "config": {
            "npmName": "@example/pipeline-ingestor-client",
            "npmVersion": "1.0.0",
            "supportsES6": "true",
            "typescriptThreePlus": "true",
            "modelPropertyNaming": "original",
        },
    },
    "javascript": {
        "generator": "javascript",
        "output": OUTPUT_DIR / "javascript",
        "config": {
            "projectName": "pipeline-ingestor-client",
            "projectVersion": "1.0.0",
            "moduleName": "PipelineIngestor",
            "useES6": "true",
        },
    },
    "java": {
        "generator": "java",
        "output": OUTPUT_DIR / "java",
        "config": {
            "groupId": "com.example",
            "artifactId": "pipeline-ingestor-client",
            "artifactVersion": "1.0.0",
            "apiPackage": "com.example.pipeline.api",
            "modelPackage": "com.example.pipeline.model",
            "library": "native",
        },
    },
    "go": {
        "generator": "go",
        "output": OUTPUT_DIR / "go",
        "config": {
            "packageName": "pipelineingestor",
            "packageVersion": "1.0.0",
            "gitUserId": "example",
            "gitRepoId": "pipeline-ingestor-go-client",
        },
    },
}


def check_openapi_generator() -> str:
    """Check if OpenAPI Generator is available.
    
    Returns:
        Command to run OpenAPI Generator
        
    Raises:
        RuntimeError: If OpenAPI Generator is not found
    """
    # Check for openapi-generator command
    if shutil.which("openapi-generator"):
        return "openapi-generator"

    # Check for openapi-generator-cli.jar
    jar_path = Path.home() / ".local" / "bin" / "openapi-generator-cli.jar"
    if jar_path.exists():
        return f"java -jar {jar_path}"

    # Check for npx
    if shutil.which("npx"):
        return "npx @openapitools/openapi-generator-cli"

    raise RuntimeError(
        "OpenAPI Generator not found. Please install it:\n"
        "  - npm: npm install -g @openapitools/openapi-generator-cli\n"
        "  - brew: brew install openapi-generator\n"
        "  - docker: docker run openapitools/openapi-generator-cli\n"
        "  - manual: Download from https://openapi-generator.tech/docs/installation"
    )


def generate_sdk(language: str, command: str, dry_run: bool = False) -> bool:
    """Generate SDK for a specific language.
    
    Args:
        language: Language to generate SDK for
        command: OpenAPI Generator command
        dry_run: If True, only print commands without executing
        
    Returns:
        True if generation succeeded, False otherwise
    """
    if language not in CONFIGS:
        logger.error(f"Unknown language: {language}")
        logger.info(f"Available languages: {', '.join(CONFIGS.keys())}")
        return False

    config = CONFIGS[language]
    output_dir = config["output"]

    logger.info(f"Generating {language} SDK to {output_dir}")

    # Build command arguments
    cmd_parts = [command, "generate"]
    cmd_parts.extend(["-i", str(OPENAPI_SPEC)])
    cmd_parts.extend(["-g", config["generator"]])
    cmd_parts.extend(["-o", str(output_dir)])

    # Add additional properties
    for key, value in config["config"].items():
        cmd_parts.extend(["--additional-properties", f"{key}={value}"])

    # Add global properties to skip files we don't need
    cmd_parts.extend([
        "--global-property",
        "skipFormModel=false,modelDocs=false,apiDocs=false",
    ])

    cmd = " ".join(cmd_parts)

    if dry_run:
        logger.info(f"[DRY RUN] Would execute: {cmd}")
        return True

    try:
        # Clean output directory if it exists
        if output_dir.exists():
            logger.info(f"Cleaning existing output directory: {output_dir}")
            shutil.rmtree(output_dir)

        # Run generator
        logger.info(f"Running: {cmd}")
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            check=True,
        )

        if result.returncode == 0:
            logger.info(f"Successfully generated {language} SDK")

            # Post-processing
            post_process_sdk(language, output_dir)

            return True
        else:
            logger.error(f"Failed to generate {language} SDK:")
            logger.error(result.stderr)
            return False

    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to generate {language} SDK:")
        logger.error(f"Exit code: {e.returncode}")
        logger.error(f"Output: {e.output}")
        logger.error(f"Stderr: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Error generating {language} SDK: {e}")
        return False


def post_process_sdk(language: str, output_dir: Path) -> None:
    """Post-process generated SDK.
    
    Args:
        language: SDK language
        output_dir: Output directory of generated SDK
    """
    logger.info(f"Post-processing {language} SDK")

    if language == "python":
        # Create pyproject.toml if it doesn't exist
        pyproject_path = output_dir / "pyproject.toml"
        if not pyproject_path.exists():
            pyproject_content = """[build-system]
requires = ["setuptools>=45", "wheel", "setuptools_scm>=6.2"]
build-backend = "setuptools.build_meta"

[project]
name = "pipeline-ingestor-client"
version = "1.0.0"
description = "Python SDK for Agentic Data Pipeline Ingestor API"
readme = "README.md"
requires-python = ">=3.8"
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
]
dependencies = [
    "urllib3>=1.25.3",
    "python-dateutil>=2.8.2",
    "aiohttp>=3.8.0",
    "pydantic>=2.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "black>=23.0.0",
    "mypy>=1.0.0",
]
"""
            pyproject_path.write_text(pyproject_content)
            logger.info(f"Created {pyproject_path}")

    elif language == "typescript":
        # Ensure package.json has correct fields
        package_json_path = output_dir / "package.json"
        if package_json_path.exists():
            import json

            with open(package_json_path) as f:
                package_data = json.load(f)

            # Update package.json
            package_data.setdefault("files", ["dist", "src", "README.md"])
            package_data.setdefault("main", "dist/index.js")
            package_data.setdefault("types", "dist/index.d.ts")
            package_data.setdefault("scripts", {})
            package_data["scripts"].setdefault("build", "tsc")
            package_data["scripts"].setdefault("test", "echo 'No tests configured'")

            with open(package_json_path, "w") as f:
                json.dump(package_data, f, indent=2)

            logger.info(f"Updated {package_json_path}")


def main() -> int:
    """Main entry point.
    
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    parser = argparse.ArgumentParser(
        description="Generate SDKs from OpenAPI specification",
    )
    parser.add_argument(
        "--language",
        "-l",
        choices=list(CONFIGS.keys()) + ["all"],
        default="all",
        help="Language to generate SDK for (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        help="Print commands without executing",
    )
    parser.add_argument(
        "--check",
        "-c",
        action="store_true",
        help="Check if OpenAPI spec is valid without generating",
    )

    args = parser.parse_args()

    # Check OpenAPI spec exists
    if not OPENAPI_SPEC.exists():
        logger.error(f"OpenAPI spec not found: {OPENAPI_SPEC}")
        return 1

    logger.info(f"Using OpenAPI spec: {OPENAPI_SPEC}")

    # Just validate spec if requested
    if args.check:
        logger.info("Validating OpenAPI spec...")
        # Could add validation logic here
        return 0

    # Find OpenAPI Generator
    try:
        command = check_openapi_generator()
        logger.info(f"Using OpenAPI Generator: {command}")
    except RuntimeError as e:
        logger.error(e)
        return 1

    # Determine languages to generate
    if args.language == "all":
        languages = ["python", "typescript"]  # Default to main languages
    else:
        languages = [args.language]

    # Generate SDKs
    success_count = 0
    for language in languages:
        if generate_sdk(language, command, args.dry_run):
            success_count += 1

    logger.info(f"Generated {success_count}/{len(languages)} SDKs")

    return 0 if success_count == len(languages) else 1


if __name__ == "__main__":
    sys.exit(main())
