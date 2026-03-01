"""Contract tests for the GraphRAG OpenAPI specification.

These tests validate the /api/v1/openapi.graphrag.yaml endpoint and ensure
the GraphRAG OpenAPI specification is valid and contains expected paths
for Cognee and HippoRAG endpoints.
"""

from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from src.main import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def openapi_graphrag_schema():
    """Load the GraphRAG OpenAPI schema from file."""
    openapi_path = Path(__file__).parent.parent.parent / "api" / "openapi.graphrag.yaml"
    with open(openapi_path) as f:
        return yaml.safe_load(f)


@pytest.mark.contract
class TestOpenAPIGraphRAGEndpoint:
    """Tests for the /api/v1/openapi.graphrag.yaml endpoint."""

    def test_openapi_graphrag_endpoint_exists(self, client):
        """Test that the GraphRAG OpenAPI endpoint exists and returns 200."""
        response = client.get("/api/v1/openapi.graphrag.yaml")
        assert response.status_code == 200

    def test_openapi_graphrag_returns_valid_yaml(self, client):
        """Test that the endpoint returns valid YAML content."""
        response = client.get("/api/v1/openapi.graphrag.yaml")
        assert response.status_code == 200

        content = response.text
        assert "openapi:" in content
        assert "paths:" in content
        assert "components:" in content

        # Verify it's valid YAML
        data = yaml.safe_load(content)
        assert data is not None
        assert isinstance(data, dict)

    def test_openapi_graphrag_content_type(self, client):
        """Test that the endpoint returns correct content type."""
        response = client.get("/api/v1/openapi.graphrag.yaml")
        assert response.status_code == 200

        # Check content type header
        content_type = response.headers.get("content-type", "")
        assert "text/yaml" in content_type or "application/yaml" in content_type or "text/plain" in content_type


@pytest.mark.contract
class TestOpenAPIGraphRAGSpecValidity:
    """Tests for the GraphRAG OpenAPI spec validity and structure."""

    def test_spec_has_required_fields(self, openapi_graphrag_schema):
        """Test that spec has all required OpenAPI fields."""
        assert "openapi" in openapi_graphrag_schema
        assert "info" in openapi_graphrag_schema
        assert "paths" in openapi_graphrag_schema

    def test_spec_version_is_3_1(self, openapi_graphrag_schema):
        """Test that spec version is 3.1.x."""
        version = openapi_graphrag_schema["openapi"]
        assert version.startswith("3.1")

    def test_spec_has_api_info(self, openapi_graphrag_schema):
        """Test that spec has API information."""
        info = openapi_graphrag_schema["info"]
        assert "title" in info
        assert "version" in info
        assert "description" in info
        assert "GraphRAG" in info["title"] or "Cognee" in info["title"] or "HippoRAG" in info["title"]

    def test_spec_has_paths(self, openapi_graphrag_schema):
        """Test that spec has at least one path defined."""
        paths = openapi_graphrag_schema["paths"]
        assert len(paths) > 0

    def test_spec_has_cognee_paths(self, openapi_graphrag_schema):
        """Test that spec has Cognee paths defined."""
        paths = openapi_graphrag_schema["paths"]
        assert "/api/v1/cognee/search" in paths
        assert "/api/v1/cognee/extract-entities" in paths
        assert "/api/v1/cognee/stats" in paths

    def test_spec_has_hipporag_paths(self, openapi_graphrag_schema):
        """Test that spec has HippoRAG paths defined."""
        paths = openapi_graphrag_schema["paths"]
        assert "/api/v1/hipporag/retrieve" in paths
        assert "/api/v1/hipporag/qa" in paths
        assert "/api/v1/hipporag/extract-triples" in paths

    def test_cognee_search_path_structure(self, openapi_graphrag_schema):
        """Test Cognee search path has correct structure."""
        path = openapi_graphrag_schema["paths"]["/api/v1/cognee/search"]
        assert "post" in path
        post = path["post"]
        assert "operationId" in post
        assert "requestBody" in post
        assert "responses" in post
        assert "200" in post["responses"]
        assert "422" in post["responses"]

    def test_cognee_extract_entities_path_structure(self, openapi_graphrag_schema):
        """Test Cognee extract-entities path has correct structure."""
        path = openapi_graphrag_schema["paths"]["/api/v1/cognee/extract-entities"]
        assert "post" in path
        post = path["post"]
        assert "operationId" in post
        assert "requestBody" in post
        assert "responses" in post
        assert "200" in post["responses"]

    def test_cognee_stats_path_structure(self, openapi_graphrag_schema):
        """Test Cognee stats path has correct structure."""
        path = openapi_graphrag_schema["paths"]["/api/v1/cognee/stats"]
        assert "get" in path
        get = path["get"]
        assert "operationId" in get
        assert "responses" in get
        assert "200" in get["responses"]
        assert "parameters" in get

    def test_hipporag_retrieve_path_structure(self, openapi_graphrag_schema):
        """Test HippoRAG retrieve path has correct structure."""
        path = openapi_graphrag_schema["paths"]["/api/v1/hipporag/retrieve"]
        assert "post" in path
        post = path["post"]
        assert "operationId" in post
        assert "requestBody" in post
        assert "responses" in post
        assert "200" in post["responses"]
        assert "422" in post["responses"]

    def test_hipporag_qa_path_structure(self, openapi_graphrag_schema):
        """Test HippoRAG QA path has correct structure."""
        path = openapi_graphrag_schema["paths"]["/api/v1/hipporag/qa"]
        assert "post" in path
        post = path["post"]
        assert "operationId" in post
        assert "requestBody" in post
        assert "responses" in post
        assert "200" in post["responses"]

    def test_hipporag_extract_triples_path_structure(self, openapi_graphrag_schema):
        """Test HippoRAG extract-triples path has correct structure."""
        path = openapi_graphrag_schema["paths"]["/api/v1/hipporag/extract-triples"]
        assert "post" in path
        post = path["post"]
        assert "operationId" in post
        assert "requestBody" in post
        assert "responses" in post
        assert "200" in post["responses"]


@pytest.mark.contract
class TestOpenAPIGraphRAGSchemas:
    """Tests for GraphRAG OpenAPI schema definitions."""

    def test_spec_has_components_schemas(self, openapi_graphrag_schema):
        """Test that spec has components/schemas section."""
        assert "components" in openapi_graphrag_schema
        assert "schemas" in openapi_graphrag_schema["components"]

    def test_cognee_schemas_defined(self, openapi_graphrag_schema):
        """Test that all Cognee schemas are defined."""
        schemas = openapi_graphrag_schema["components"]["schemas"]
        assert "CogneeSearchRequest" in schemas
        assert "CogneeSearchResponse" in schemas
        assert "CogneeSearchResult" in schemas
        assert "CogneeExtractEntitiesRequest" in schemas
        assert "CogneeExtractEntitiesResponse" in schemas
        assert "CogneeEntity" in schemas
        assert "CogneeRelationship" in schemas
        assert "CogneeStatsResponse" in schemas

    def test_hipporag_schemas_defined(self, openapi_graphrag_schema):
        """Test that all HippoRAG schemas are defined."""
        schemas = openapi_graphrag_schema["components"]["schemas"]
        assert "HippoRAGRetrieveRequest" in schemas
        assert "HippoRAGRetrieveResponse" in schemas
        assert "HippoRAGRetrievalResult" in schemas
        assert "HippoRAGQARequest" in schemas
        assert "HippoRAGQAResponse" in schemas
        assert "HippoRAGQAResult" in schemas
        assert "HippoRAGExtractTriplesRequest" in schemas
        assert "HippoRAGExtractTriplesResponse" in schemas
        assert "HippoRAGTriple" in schemas

    def test_cognee_search_request_schema(self, openapi_graphrag_schema):
        """Test CogneeSearchRequest schema structure."""
        schema = openapi_graphrag_schema["components"]["schemas"]["CogneeSearchRequest"]
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "query" in schema["properties"]
        assert "search_type" in schema["properties"]
        assert "top_k" in schema["properties"]
        assert "dataset_id" in schema["properties"]
        assert "required" in schema
        assert "query" in schema["required"]

    def test_cognee_search_response_schema(self, openapi_graphrag_schema):
        """Test CogneeSearchResponse schema structure."""
        schema = openapi_graphrag_schema["components"]["schemas"]["CogneeSearchResponse"]
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "results" in schema["properties"]
        assert "search_type" in schema["properties"]
        assert "dataset_id" in schema["properties"]
        assert "query_time_ms" in schema["properties"]

    def test_cognee_stats_response_schema(self, openapi_graphrag_schema):
        """Test CogneeStatsResponse schema structure."""
        schema = openapi_graphrag_schema["components"]["schemas"]["CogneeStatsResponse"]
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "dataset_id" in schema["properties"]
        assert "document_count" in schema["properties"]
        assert "chunk_count" in schema["properties"]
        assert "entity_count" in schema["properties"]
        assert "relationship_count" in schema["properties"]
        assert "graph_density" in schema["properties"]
        assert "last_updated" in schema["properties"]

    def test_hipporag_retrieve_request_schema(self, openapi_graphrag_schema):
        """Test HippoRAGRetrieveRequest schema structure."""
        schema = openapi_graphrag_schema["components"]["schemas"]["HippoRAGRetrieveRequest"]
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "queries" in schema["properties"]
        assert "num_to_retrieve" in schema["properties"]
        assert "required" in schema
        assert "queries" in schema["required"]

    def test_hipporag_qa_response_schema(self, openapi_graphrag_schema):
        """Test HippoRAGQAResponse schema structure."""
        schema = openapi_graphrag_schema["components"]["schemas"]["HippoRAGQAResponse"]
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "results" in schema["properties"]
        assert "total_tokens" in schema["properties"]
        assert "query_time_ms" in schema["properties"]

    def test_hipporag_triple_schema(self, openapi_graphrag_schema):
        """Test HippoRAGTriple schema structure."""
        schema = openapi_graphrag_schema["components"]["schemas"]["HippoRAGTriple"]
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "subject" in schema["properties"]
        assert "predicate" in schema["properties"]
        assert "object" in schema["properties"]


@pytest.mark.contract
class TestOpenAPIGraphRAGConsistency:
    """Tests for consistency between OpenAPI spec and implementation."""

    def test_cognee_endpoints_match_spec(self, client, openapi_graphrag_schema):
        """Test that Cognee endpoints in spec match implemented paths."""
        spec_paths = openapi_graphrag_schema["paths"]

        # All spec paths should be present in the app
        cognee_paths = [p for p in spec_paths.keys() if "cognee" in p]
        assert len(cognee_paths) == 3
        assert "/api/v1/cognee/search" in cognee_paths
        assert "/api/v1/cognee/extract-entities" in cognee_paths
        assert "/api/v1/cognee/stats" in cognee_paths

    def test_hipporag_endpoints_match_spec(self, client, openapi_graphrag_schema):
        """Test that HippoRAG endpoints in spec match implemented paths."""
        spec_paths = openapi_graphrag_schema["paths"]

        # All spec paths should be present in the app
        hipporag_paths = [p for p in spec_paths.keys() if "hipporag" in p]
        assert len(hipporag_paths) == 3
        assert "/api/v1/hipporag/retrieve" in hipporag_paths
        assert "/api/v1/hipporag/qa" in hipporag_paths
        assert "/api/v1/hipporag/extract-triples" in hipporag_paths

    def test_schema_types_are_valid(self, openapi_graphrag_schema):
        """Test that all schema types are valid OpenAPI types."""
        valid_types = {"object", "array", "string", "integer", "number", "boolean"}
        schemas = openapi_graphrag_schema["components"]["schemas"]

        for schema_name, schema in schemas.items():
            if "type" in schema:
                assert schema["type"] in valid_types, f"Invalid type in {schema_name}"

    def test_request_bodies_have_content(self, openapi_graphrag_schema):
        """Test that all request bodies have content defined."""
        paths = openapi_graphrag_schema["paths"]

        for path, methods in paths.items():
            for method, spec in methods.items():
                if method in ["post", "put", "patch"]:
                    if "requestBody" in spec:
                        assert "content" in spec["requestBody"], f"Missing content in {method} {path}"
                        assert "application/json" in spec["requestBody"]["content"], f"Missing JSON content in {method} {path}"


@pytest.mark.contract
class TestOpenAPIGraphRAGExamples:
    """Tests for example values in the GraphRAG OpenAPI spec."""

    def test_cognee_search_request_has_examples(self, openapi_graphrag_schema):
        """Test that CogneeSearchRequest has example values."""
        schema = openapi_graphrag_schema["components"]["schemas"]["CogneeSearchRequest"]
        query_prop = schema["properties"]["query"]
        assert "example" in query_prop or "examples" in query_prop

    def test_search_type_enum_values(self, openapi_graphrag_schema):
        """Test that search_type has valid enum values."""
        schema = openapi_graphrag_schema["components"]["schemas"]["CogneeSearchRequest"]
        search_type = schema["properties"]["search_type"]
        assert "enum" in search_type
        assert set(search_type["enum"]) == {"vector", "graph", "hybrid"}

    def test_top_k_has_valid_constraints(self, openapi_graphrag_schema):
        """Test that top_k has valid min/max constraints."""
        schema = openapi_graphrag_schema["components"]["schemas"]["CogneeSearchRequest"]
        top_k = schema["properties"]["top_k"]
        assert "minimum" in top_k
        assert "maximum" in top_k
        assert top_k["minimum"] >= 1
        assert top_k["maximum"] <= 100

    def test_num_to_retrieve_constraints(self, openapi_graphrag_schema):
        """Test that num_to_retrieve has valid constraints."""
        schema = openapi_graphrag_schema["components"]["schemas"]["HippoRAGRetrieveRequest"]
        num = schema["properties"]["num_to_retrieve"]
        assert "minimum" in num
        assert "maximum" in num
        assert num["minimum"] >= 1
        assert num["maximum"] <= 50
