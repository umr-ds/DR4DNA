# -*- coding: utf-8 -*-
"""
Callback dispatcher for DR4DNA application.

Provides a clean dispatch mechanism for routing callback events to handlers.
"""

import typing
from dataclasses import dataclass
from logger import get_logger

logger = get_logger(__name__)


@dataclass
class CallbackRoute:
    """Represents a callback route mapping trigger IDs to handler methods."""
    trigger_ids: typing.List[str]
    handler_method: str
    description: str = ""


class CallbackDispatcher:
    """
    Dispatches callback events to appropriate handler methods.
    
    This class replaces long if-elif chains with a clean dispatch table pattern.
    
    Example:
        >>> dispatcher = CallbackDispatcher()
        >>> dispatcher.register_route(CallbackRoute(
        ...     trigger_ids=["analyze-button"],
        ...     handler_method="handle_analyze_button"
        ... ))
        >>> result = dispatcher.dispatch("analyze-button", handler_instance)
    """
    
    def __init__(self):
        self._routes: typing.List[CallbackRoute] = []
    
    def register_route(self, route: CallbackRoute) -> None:
        """
        Register a callback route.
        
        Args:
            route: The callback route to register
        """
        self._routes.append(route)
    
    def register_routes(self, routes: typing.List[CallbackRoute]) -> None:
        """
        Register multiple callback routes.
        
        Args:
            routes: List of callback routes to register
        """
        self._routes.extend(routes)
    
    def dispatch(self, trigger_id: str, handler: typing.Any, *args, **kwargs) -> typing.Any:
        """
        Dispatch a callback event to the appropriate handler method.
        
        Args:
            trigger_id: The ID of the triggered element
            handler: The handler instance containing the handler methods
            *args: Positional arguments to pass to the handler method
            **kwargs: Keyword arguments to pass to the handler method
            
        Returns:
            The result from the handler method, or None if no route matched
            
        Raises:
            AttributeError: If the handler method doesn't exist
        """
        for route in self._routes:
            if trigger_id in route.trigger_ids:
                handler_method = getattr(handler, route.handler_method, None)
                if handler_method is None:
                    logger.error(f"Handler method '{route.handler_method}' not found")
                    raise AttributeError(f"Handler method '{route.handler_method}' not found")
                
                logger.debug(f"Dispatching '{trigger_id}' to '{route.handler_method}'")
                return handler_method(*args, **kwargs)
        
        logger.warning(f"No route found for trigger_id: {trigger_id}")
        return None
    
    def get_route_for_trigger(self, trigger_id: str) -> typing.Optional[CallbackRoute]:
        """
        Get the route matching a trigger ID.
        
        Args:
            trigger_id: The trigger ID to look up
            
        Returns:
            The matching CallbackRoute, or None if not found
        """
        for route in self._routes:
            if trigger_id in route.trigger_ids:
                return route
        return None
    
    def list_routes(self) -> typing.List[typing.Dict[str, str]]:
        """
        List all registered routes.
        
        Returns:
            List of route information dictionaries
        """
        return [
            {
                "trigger_ids": route.trigger_ids,
                "handler_method": route.handler_method,
                "description": route.description
            }
            for route in self._routes
        ]
