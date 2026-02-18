"""Unit tests for RBAC (Role-Based Access Control)."""

from uuid import uuid4

import pytest

from src.auth.base import Permission, User
from src.auth.rbac import (
    RBACManager,
    Role,
    RoleDefinition,
    check_permission,
    get_rbac_manager,
    require_admin,
    require_operator,
)


class TestRole:
    """Tests for Role enum."""

    def test_role_values(self):
        """Test role values."""
        assert Role.ADMIN == "admin"
        assert Role.OPERATOR == "operator"
        assert Role.DEVELOPER == "developer"
        assert Role.VIEWER == "viewer"


class TestRoleDefinition:
    """Tests for RoleDefinition."""

    def test_role_has_permission(self):
        """Test checking role permission."""
        role = RoleDefinition(
            name="test",
            display_name="Test",
            description="Test role",
            permissions={Permission.READ, Permission.CREATE},
        )

        assert role.has_permission(Permission.READ) is True
        assert role.has_permission(Permission.DELETE) is False

    def test_admin_role_has_all_permissions(self):
        """Test admin role has all permissions."""
        role = RoleDefinition(
            name="admin",
            display_name="Admin",
            description="Admin role",
            permissions={Permission.ADMIN},
        )

        assert role.has_permission(Permission.READ) is True
        assert role.has_permission(Permission.DELETE) is True
        assert role.has_permission(Permission.MANAGE_USERS) is True

    def test_role_to_dict(self):
        """Test converting role to dict."""
        role = RoleDefinition(
            name="operator",
            display_name="Operator",
            description="Can operate",
            permissions={Permission.READ, Permission.CREATE},
        )

        d = role.to_dict()

        assert d["name"] == "operator"
        assert d["display_name"] == "Operator"
        assert "read" in d["permissions"]
        assert "create" in d["permissions"]


class TestRBACManager:
    """Tests for RBACManager."""

    def test_get_standard_role(self):
        """Test getting a standard role."""
        rbac = RBACManager()
        role = rbac.get_role("admin")

        assert role is not None
        assert role.name == "admin"
        assert Permission.ADMIN in role.permissions

    def test_get_nonexistent_role(self):
        """Test getting nonexistent role."""
        rbac = RBACManager()
        role = rbac.get_role("nonexistent")

        assert role is None

    def test_list_roles(self):
        """Test listing all roles."""
        rbac = RBACManager()
        roles = rbac.list_roles()

        role_names = [r.name for r in roles]
        assert "admin" in role_names
        assert "operator" in role_names
        assert "developer" in role_names
        assert "viewer" in role_names

    def test_check_permission_admin(self):
        """Test admin always has permission."""
        rbac = RBACManager()
        user = User(id=uuid4(), roles=["admin"])

        assert rbac.check_permission(user, "jobs", "delete") is True
        assert rbac.check_permission(user, "anything", "anything") is True

    def test_check_permission_operator(self):
        """Test operator permissions."""
        rbac = RBACManager()
        user = User(id=uuid4(), role="operator")

        assert rbac.check_permission(user, "jobs", "read") is True
        assert rbac.check_permission(user, "jobs", "create") is True
        assert rbac.check_permission(user, "jobs", "cancel") is True
        assert rbac.check_permission(user, "jobs", "delete") is False

    def test_check_permission_developer(self):
        """Test developer permissions."""
        rbac = RBACManager()
        user = User(id=uuid4(), role="developer")

        assert rbac.check_permission(user, "jobs", "read") is True
        assert rbac.check_permission(user, "jobs", "submit") is True  # submit uses CREATE_JOBS
        assert rbac.check_permission(user, "jobs", "delete") is False

    def test_check_permission_viewer(self):
        """Test viewer permissions."""
        rbac = RBACManager()
        user = User(id=uuid4(), role="viewer")

        assert rbac.check_permission(user, "jobs", "read") is True
        assert rbac.check_permission(user, "jobs", "create") is False

    def test_check_permission_unknown_resource(self):
        """Test unknown resource defaults to deny."""
        rbac = RBACManager()
        user = User(id=uuid4(), role="operator")

        assert rbac.check_permission(user, "unknown_resource", "action") is False

    def test_get_user_permissions(self):
        """Test getting all user permissions."""
        rbac = RBACManager()
        user = User(
            id=uuid4(),
            role="operator",
            permissions=["view_audit"],
        )

        perms = rbac.get_user_permissions(user)

        assert Permission.READ in perms
        assert Permission.CREATE in perms
        assert Permission.VIEW_AUDIT in perms

    def test_create_custom_role(self):
        """Test creating custom role."""
        rbac = RBACManager()
        role = rbac.create_custom_role(
            name="custom",
            display_name="Custom Role",
            description="A custom role",
            permissions=[Permission.READ, Permission.CREATE],
        )

        assert role.name == "custom"
        assert rbac.get_role("custom") == role

    def test_create_custom_role_with_inheritance(self):
        """Test creating custom role with inheritance."""
        rbac = RBACManager()
        role = rbac.create_custom_role(
            name="super_operator",
            display_name="Super Operator",
            description="Extended operator",
            permissions=[Permission.VIEW_AUDIT],
            inherits_from="operator",
        )

        assert Permission.READ in role.permissions
        assert Permission.CREATE in role.permissions  # From operator
        assert Permission.VIEW_AUDIT in role.permissions  # Own

    def test_delete_custom_role(self):
        """Test deleting custom role."""
        rbac = RBACManager()
        rbac.create_custom_role(
            name="temp",
            display_name="Temp",
            description="Temporary",
            permissions=[Permission.READ],
        )

        result = rbac.delete_custom_role("temp")

        assert result is True
        assert rbac.get_role("temp") is None

    def test_cannot_delete_standard_role(self):
        """Test cannot delete standard roles."""
        rbac = RBACManager()

        result = rbac.delete_custom_role("admin")

        assert result is False

    def test_list_resources(self):
        """Test listing managed resources."""
        rbac = RBACManager()
        resources = rbac.list_resources()

        assert "jobs" in resources
        assert "sources" in resources
        assert "audit" in resources


class TestRBACConvenienceFunctions:
    """Tests for RBAC convenience functions."""

    def test_get_rbac_manager_singleton(self):
        """Test RBAC manager is singleton."""
        manager1 = get_rbac_manager()
        manager2 = get_rbac_manager()

        assert manager1 is manager2

    def test_check_permission_function(self):
        """Test check_permission convenience function."""
        user = User(id=uuid4(), role="admin")

        assert check_permission(user, "jobs", "delete") is True

    def test_require_admin(self):
        """Test require_admin function."""
        admin_user = User(id=uuid4(), roles=["admin"])
        operator_user = User(id=uuid4(), role="operator")

        assert require_admin(admin_user) is True
        assert require_admin(operator_user) is False

    def test_require_operator(self):
        """Test require_operator function."""
        admin_user = User(id=uuid4(), roles=["admin"])
        operator_user = User(id=uuid4(), role="operator")
        viewer_user = User(id=uuid4(), role="viewer")

        assert require_operator(admin_user) is True
        assert require_operator(operator_user) is True
        assert require_operator(viewer_user) is False
