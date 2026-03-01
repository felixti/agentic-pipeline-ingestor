# Tasks: Implement Cognee and HippoRAG API Endpoints

## Task List

### Phase 1: Setup and Models
- [ ] Task 1.1: Create Pydantic models for Cognee endpoints
- [ ] Task 1.2: Create Pydantic models for HippoRAG endpoints

### Phase 2: Cognee Routes
- [ ] Task 2.1: Implement POST /api/v1/cognee/search endpoint
- [ ] Task 2.2: Implement POST /api/v1/cognee/extract-entities endpoint
- [ ] Task 2.3: Implement GET /api/v1/cognee/stats endpoint

### Phase 3: HippoRAG Routes
- [ ] Task 3.1: Implement POST /api/v1/hipporag/retrieve endpoint
- [ ] Task 3.2: Implement POST /api/v1/hipporag/qa endpoint
- [ ] Task 3.3: Implement POST /api/v1/hipporag/extract-triples endpoint

### Phase 4: Integration
- [ ] Task 4.1: Register routes in src/api/routes/__init__.py
- [ ] Task 4.2: Add routes to main.py
- [ ] Task 4.3: Implement /api/v1/openapi.graphrag.yaml endpoint

### Phase 5: Testing
- [ ] Task 5.1: Create unit tests for Cognee endpoints
- [ ] Task 5.2: Create unit tests for HippoRAG endpoints
- [ ] Task 5.3: Create contract tests for OpenAPI compliance
- [ ] Task 5.4: Run full test suite and fix any issues

### Phase 6: Validation
- [ ] Task 6.1: Verify all 6 endpoints return 200 OK
- [ ] Task 6.2: Verify openapi.graphrag.yaml endpoint works
- [ ] Task 6.3: Verify contract tests pass

## Task Details

### Task 1.1: Cognee Models
**Owner**: backend-developer
**Dependencies**: None
**Acceptance Criteria**:
- All Cognee request/response models defined
- Models match openapi.graphrag.yaml spec
- Proper validation and examples

### Task 1.2: HippoRAG Models
**Owner**: backend-developer
**Dependencies**: None
**Acceptance Criteria**:
- All HippoRAG request/response models defined
- Models match openapi.graphrag.yaml spec
- Proper validation and examples

### Task 2.1-2.3: Cognee Routes
**Owner**: backend-developer
**Dependencies**: Task 1.1
**Acceptance Criteria**:
- Each endpoint returns proper response
- Integrates with CogneeLocalGraph plugin
- Proper error handling

### Task 3.1-3.3: HippoRAG Routes
**Owner**: backend-developer
**Dependencies**: Task 1.2
**Acceptance Criteria**:
- Each endpoint returns proper response
- Integrates with HippoRAGDestination plugin
- Proper error handling

### Task 4.1-4.3: Integration
**Owner**: backend-developer
**Dependencies**: Tasks 2.1-3.3
**Acceptance Criteria**:
- Routes registered and accessible
- openapi.graphrag.yaml endpoint serves spec
- No breaking changes to existing routes

### Task 5.1-5.4: Testing
**Owner**: tester-agent
**Dependencies**: Task 4.3
**Acceptance Criteria**:
- Unit tests cover all endpoints
- Contract tests pass
- Full test suite passes

### Task 6.1-6.3: Validation
**Owner**: qa-agent
**Dependencies**: Task 5.4
**Acceptance Criteria**:
- All endpoints verified working
- OpenAPI spec compliance verified
- No regressions
