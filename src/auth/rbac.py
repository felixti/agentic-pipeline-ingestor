"""Role-Based Access Control (RBAC) system.

This module provides role-based access control with support for
enterprise-grade authorization patterns.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from uuid import UUID

from src.auth.base import Permission, User


class Role(str, Enum):
    """Pre-defined roles in the system.
    
    The system supports 4 primary roles as specified in Section 9.2:
    - admin: Full system access
    - operator: Create, read, cancel, retry jobs
    - developer: Create jobs, read sources/destinations
    - viewer: Read-only access
    """
    ADMIN = "admin"
    OPERATOR = "operator"
    DEVELOPER = "developer"
    VIEWER = "viewer"


@dataclass
class RoleDefinition:
    """Definition of a role and its permissions.
    
    Attributes:
        name: Role identifier
        display_name: Human-readable name
        description: Role description
        permissions: Set of permissions granted
        inherits_from: Optional parent role to inherit permissions from
    """
    name: str
    display_name: str
    description: str
    permissions: Set[Permission]
    inherits_from: Optional[str] = None
    
    def has_permission(self, permission: Permission) -> bool:
        """Check if role has a specific permission."""
        return permission in self.permissions or Permission.ADMIN in self.permissions
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "permissions": [p.value for p in self.permissions],
            "inherits_from": self.inherits_from,
        }


class RBACManager:
    """Role-Based Access Control manager.
    
    This class manages roles, permissions, and access control decisions.
    It supports the 4 standard roles defined in the spec.
    
    Role Permissions (from spec Section 9.2):
    - admin: All operations (Permission.ALL)
    - operator: READ | CREATE | CANCEL | RETRY
    - developer: READ | CREATE_JOBS | READ_SOURCES
    - viewer: READ_ONLY
    
    Example:
        rbac = RBACManager()
        
        # Check if user can perform action
        if rbac.check_permission(user, "jobs", "create"):
            # Allow action
            pass
        
        # Get user's permissions
        permissions = rbac.get_user_permissions(user)
    """
    
    # Standard role definitions
    ROLES: Dict[str, RoleDefinition] = {
        Role.ADMIN.value: RoleDefinition(
            name=Role.ADMIN.value,
            display_name="Administrator",
            description="Full system access with all permissions",
            permissions={
                Permission.ADMIN,
                Permission.READ,
                Permission.CREATE,
                Permission.UPDATE,
                Permission.DELETE,
                Permission.CANCEL,
                Permission.RETRY,
                Permission.CREATE_JOBS,
                Permission.READ_SOURCES,
                Permission.MANAGE_CONFIG,
                Permission.MANAGE_USERS,
                Permission.VIEW_AUDIT,
                Permission.EXPORT_DATA,
            },
        ),
        Role.OPERATOR.value: RoleDefinition(
            name=Role.OPERATOR.value,
            display_name="Operator",
            description="Can create, read, cancel, and retry jobs",
            permissions={
                Permission.READ,
                Permission.CREATE,
                Permission.CANCEL,
                Permission.RETRY,
                Permission.CREATE_JOBS,
            },
        ),
        Role.DEVELOPER.value: RoleDefinition(
            name=Role.DEVELOPER.value,
            display_name="Developer",
            description="Can create jobs and read sources/destinations",
            permissions={
                Permission.READ,
                Permission.CREATE_JOBS,
                Permission.READ_SOURCES,
            },
        ),
        Role.VIEWER.value: RoleDefinition(
            name=Role.VIEWER.value,
            display_name="Viewer",
            description="Read-only access to jobs and resources",
            permissions={
                Permission.READ,
            },
        ),
    }
    
    # Resource-action to permission mapping
    RESOURCE_ACTION_PERMISSIONS: Dict[str, Dict[str, Permission]] = {
        "jobs": {
            "read": Permission.READ,
            "create": Permission.CREATE,
            "update": Permission.UPDATE,
            "delete": Permission.DELETE,
            "cancel": Permission.CANCEL,
            "retry": Permission.RETRY,
            "submit": Permission.CREATE_JOBS,
        },
        "sources": {
            "read": Permission.READ_SOURCES,
            "create": Permission.MANAGE_CONFIG,
            "update": Permission.MANAGE_CONFIG,
            "delete": Permission.MANAGE_CONFIG,
            "test": Permission.READ_SOURCES,
        },
        "destinations": {
            "read": Permission.READ_SOURCES,
            "create": Permission.MANAGE_CONFIG,
            "update": Permission.MANAGE_CONFIG,
            "delete": Permission.MANAGE_CONFIG,
            "test": Permission.READ_SOURCES,
        },
        "pipelines": {
            "read": Permission.READ,
            "create": Permission.MANAGE_CONFIG,
            "update": Permission.MANAGE_CONFIG,
            "delete": Permission.MANAGE_CONFIG,
            "validate": Permission.READ,
        },
        "audit": {
            "read": Permission.VIEW_AUDIT,
            "export": Permission.EXPORT_DATA,
        },
        "users": {
            "read": Permission.MANAGE_USERS,
            "create": Permission.MANAGE_USERS,
            "update": Permission.MANAGE_USERS,
            "delete": Permission.MANAGE_USERS,
        },
        "api_keys": {
            "read": Permission.MANAGE_USERS,
            "create": Permission.MANAGE_USERS,
            "revoke": Permission.MANAGE_USERS,
        },
        "lineage": {
            "read": Permission.READ,
        },
        "dlq": {
            "read": Permission.READ,
            "retry": Permission.RETRY,
            "archive": Permission.MANAGE_CONFIG,
        },
        "system": {
            "read": Permission.READ,
            "health": Permission.READ,
            "metrics": Permission.READ,
        },
    }
    
    def __init__(self):
        """Initialize RBAC manager."""
        self._custom_roles: Dict[str, RoleDefinition] = {}
    
    def get_role(self, role_name: str) -> Optional[RoleDefinition]:
        """Get role definition by name.
        
        Args:
            role_name: Name of the role
            
        Returns:
            Role definition or None if not found
        """
        # Check standard roles
        if role_name in self.ROLES:
            return self.ROLES[role_name]
        
        # Check custom roles
        return self._custom_roles.get(role_name)
    
    def list_roles(self) -> List[RoleDefinition]:
        """List all available roles.
        
        Returns:
            List of role definitions
        """
        return list(self.ROLES.values()) + list(self._custom_roles.values())
    
    def check_permission(
        self,
        user: User,
        resource: str,
        action: str,
    ) -> bool:
        """Check if user has permission for a resource action.
        
        Args:
            user: User to check permissions for
            resource: Resource type (e.g., "jobs", "sources")
            action: Action to perform (e.g., "create", "read")
            
        Returns:
            True if user has permission
        """
        # Admin always has permission
        if Role.ADMIN.value in user.roles or user.role == Role.ADMIN.value:
            return True
        
        # Get required permission for resource-action
        required_permission = self._get_required_permission(resource, action)
        if not required_permission:
            # Unknown resource-action, deny by default
            return False
        
        # Check user's roles for permission
        for role_name in user.roles:
            role = self.get_role(role_name)
            if role and role.has_permission(required_permission):
                return True
        
        # Check primary role
        role = self.get_role(user.role)
        if role and role.has_permission(required_permission):
            return True
        
        # Check explicit permissions
        if required_permission.value in user.permissions:
            return True
        
        return False
    
    def _get_required_permission(
        self,
        resource: str,
        action: str,
    ) -> Optional[Permission]:
        """Get the required permission for a resource action.
        
        Args:
            resource: Resource type
            action: Action name
            
        Returns:
            Required permission or None
        """
        resource_perms = self.RESOURCE_ACTION_PERMISSIONS.get(resource.lower())
        if resource_perms:
            return resource_perms.get(action.lower())
        return None
    
    def get_user_permissions(self, user: User) -> Set[Permission]:
        """Get all permissions for a user.
        
        Args:
            user: User to get permissions for
            
        Returns:
            Set of all permissions
        """
        permissions: Set[Permission] = set()
        
        # Collect from all roles
        for role_name in [user.role] + user.roles:
            role = self.get_role(role_name)
            if role:
                permissions.update(role.permissions)
        
        # Add explicit permissions
        for perm_str in user.permissions:
            try:
                permissions.add(Permission(perm_str))
            except ValueError:
                pass
        
        return permissions
    
    def require_permission(self, resource: str, action: str):
        """Decorator to require permission for a function.
        
        Args:
            resource: Resource type
            action: Action to require
            
        Returns:
            Decorator function
        """
        def decorator(func):
            async def wrapper(*args, user: User = None, **kwargs):
                if user is None:
                    raise PermissionError("Authentication required")
                
                if not self.check_permission(user, resource, action):
                    raise PermissionError(
                        f"User {user.id} lacks permission to {action} {resource}"
                    )
                
                return await func(*args, user=user, **kwargs)
            
            return wrapper
        return decorator
    
    def create_custom_role(
        self,
        name: str,
        display_name: str,
        description: str,
        permissions: List[Permission],
        inherits_from: Optional[str] = None,
    ) -> RoleDefinition:
        """Create a custom role.
        
        Args:
            name: Unique role name
            display_name: Human-readable name
            description: Role description
            permissions: List of permissions
            inherits_from: Optional parent role
            
        Returns:
            Created role definition
        """
        # Inherit permissions if specified
        all_permissions = set(permissions)
        if inherits_from:
            parent = self.get_role(inherits_from)
            if parent:
                all_permissions.update(parent.permissions)
        
        role = RoleDefinition(
            name=name,
            display_name=display_name,
            description=description,
            permissions=all_permissions,
            inherits_from=inherits_from,
        )
        
        self._custom_roles[name] = role
        return role
    
    def delete_custom_role(self, name: str) -> bool:
        """Delete a custom role.
        
        Args:
            name: Role name to delete
            
        Returns:
            True if deleted, False if not found or is standard role
        """
        if name in self.ROLES:
            return False  # Cannot delete standard roles
        
        if name in self._custom_roles:
            del self._custom_roles[name]
            return True
        
        return False
    
    def get_resource_actions(self, resource: str) -> Dict[str, Permission]:
        """Get all actions available for a resource.
        
        Args:
            resource: Resource type
            
        Returns:
            Dictionary mapping actions to permissions
        """
        return self.RESOURCE_ACTION_PERMISSIONS.get(resource.lower(), {}).copy()
    
    def list_resources(self) -> List[str]:
        """List all managed resources.
        
        Returns:
            List of resource names
        """
        return list(self.RESOURCE_ACTION_PERMISSIONS.keys())


# Global RBAC manager instance
_rbac_manager: Optional[RBACManager] = None


def get_rbac_manager() -> RBACManager:
    """Get global RBAC manager instance.
    
    Returns:
        RBAC manager singleton
    """
    global _rbac_manager
    if _rbac_manager is None:
        _rbac_manager = RBACManager()
    return _rbac_manager


def check_permission(user: User, resource: str, action: str) -> bool:
    """Convenience function to check permission.
    
    Args:
        user: User to check
        resource: Resource type
        action: Action to check
        
    Returns:
        True if user has permission
    """
    return get_rbac_manager().check_permission(user, resource, action)


def require_admin(user: User) -> bool:
    """Check if user has admin role.
    
    Args:
        user: User to check
        
    Returns:
        True if user is admin
    """
    return user.has_role(Role.ADMIN.value) or Role.ADMIN.value in user.roles


def require_operator(user: User) -> bool:
    """Check if user has operator or higher role.
    
    Args:
        user: User to check
        
    Returns:
        True if user is operator or admin
    """
    return (
        user.has_role(Role.ADMIN.value) or
        user.has_role(Role.OPERATOR.value) or
        Role.ADMIN.value in user.roles or
        Role.OPERATOR.value in user.roles
    )
