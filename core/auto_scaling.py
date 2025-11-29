"""
Auto-Scaling Infrastructure System
Provides dynamic resource scaling and performance optimization
"""

import asyncio
import logging
import time
import psutil
import docker
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import threading
from collections import deque, defaultdict
import statistics


class ScalingDirection(Enum):
    """Scaling direction options"""
    UP = "up"
    DOWN = "down"
    NONE = "none"


class ResourceType(Enum):
    """Resource types for monitoring"""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    CONNECTIONS = "connections"


@dataclass
class ResourceMetrics:
    """Resource utilization metrics"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_io: Dict[str, int]
    active_connections: int
    load_average: List[float]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'cpu_percent': self.cpu_percent,
            'memory_percent': self.memory_percent,
            'disk_percent': self.disk_percent,
            'network_io': self.network_io,
            'active_connections': self.active_connections,
            'load_average': self.load_average
        }


@dataclass
class ScalingRule:
    """Auto-scaling rule configuration"""
    name: str
    resource_type: ResourceType
    threshold_up: float
    threshold_down: float
    duration_minutes: int = 5
    cooldown_minutes: int = 10
    min_instances: int = 1
    max_instances: int = 10
    scale_up_count: int = 1
    scale_down_count: int = 1
    enabled: bool = True


@dataclass
class ScalingEvent:
    """Scaling event record"""
    timestamp: datetime
    rule_name: str
    direction: ScalingDirection
    from_instances: int
    to_instances: int
    trigger_value: float
    reason: str
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class ServiceConfig:
    """Service configuration for scaling"""
    name: str
    image: str
    port: int
    environment: Dict[str, str] = field(default_factory=dict)
    volumes: List[str] = field(default_factory=list)
    networks: List[str] = field(default_factory=list)
    health_check_path: str = "/health"
    min_instances: int = 1
    max_instances: int = 5


class ResourceMonitor:
    """Monitors system resource utilization"""
    
    def __init__(self, collection_interval: int = 30):
        self.collection_interval = collection_interval
        self.logger = logging.getLogger(__name__)
        self.metrics_history: deque = deque(maxlen=1000)
        self.running = False
        self.monitor_task = None
    
    async def start_monitoring(self):
        """Start resource monitoring"""
        self.running = True
        self.monitor_task = asyncio.create_task(self._monitor_loop())
        self.logger.info("Resource monitoring started")
    
    async def stop_monitoring(self):
        """Stop resource monitoring"""
        self.running = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Resource monitoring stopped")
    
    async def _monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                metrics = await self._collect_metrics()
                self.metrics_history.append(metrics)
                await asyncio.sleep(self.collection_interval)
            except Exception as e:
                self.logger.error(f"Error collecting metrics: {e}")
                await asyncio.sleep(self.collection_interval)
    
    async def _collect_metrics(self) -> ResourceMetrics:
        """Collect current resource metrics"""
        # CPU utilization
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory utilization
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # Disk utilization
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        
        # Network I/O
        network = psutil.net_io_counters()
        network_io = {
            'bytes_sent': network.bytes_sent,
            'bytes_recv': network.bytes_recv,
            'packets_sent': network.packets_sent,
            'packets_recv': network.packets_recv
        }
        
        # Active connections
        connections = len(psutil.net_connections())
        
        # Load average
        load_avg = list(psutil.getloadavg()) if hasattr(psutil, 'getloadavg') else [0.0, 0.0, 0.0]
        
        return ResourceMetrics(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            disk_percent=disk_percent,
            network_io=network_io,
            active_connections=connections,
            load_average=load_avg
        )
    
    def get_recent_metrics(self, minutes: int = 10) -> List[ResourceMetrics]:
        """Get metrics from the last N minutes"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        return [
            metric for metric in self.metrics_history
            if metric.timestamp >= cutoff_time
        ]
    
    def get_average_metrics(self, minutes: int = 5) -> Optional[Dict[str, float]]:
        """Get average metrics over the specified time period"""
        recent_metrics = self.get_recent_metrics(minutes)
        
        if not recent_metrics:
            return None
        
        return {
            'cpu_percent': statistics.mean(m.cpu_percent for m in recent_metrics),
            'memory_percent': statistics.mean(m.memory_percent for m in recent_metrics),
            'disk_percent': statistics.mean(m.disk_percent for m in recent_metrics),
            'active_connections': statistics.mean(m.active_connections for m in recent_metrics),
            'load_average': statistics.mean(m.load_average[0] for m in recent_metrics if m.load_average)
        }


class ContainerManager:
    """Manages Docker containers for scaling"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        try:
            self.docker_client = docker.from_env()
            self.docker_available = True
        except Exception as e:
            self.logger.warning(f"Docker not available: {e}")
            self.docker_client = None
            self.docker_available = False
        
        self.managed_containers: Dict[str, List[str]] = defaultdict(list)
    
    async def scale_service(self, service_config: ServiceConfig, target_instances: int) -> bool:
        """Scale a service to the target number of instances"""
        if not self.docker_available:
            self.logger.error("Docker not available for scaling")
            return False
        
        try:
            current_containers = self._get_service_containers(service_config.name)
            current_count = len(current_containers)
            
            if target_instances > current_count:
                # Scale up
                instances_to_add = target_instances - current_count
                for i in range(instances_to_add):
                    container_name = f"{service_config.name}_{current_count + i + 1}"
                    await self._start_container(service_config, container_name)
                
            elif target_instances < current_count:
                # Scale down
                instances_to_remove = current_count - target_instances
                containers_to_stop = current_containers[-instances_to_remove:]
                for container_id in containers_to_stop:
                    await self._stop_container(container_id)
            
            self.logger.info(f"Scaled {service_config.name} from {current_count} to {target_instances} instances")
            return True
            
        except Exception as e:
            self.logger.error(f"Error scaling service {service_config.name}: {e}")
            return False
    
    def _get_service_containers(self, service_name: str) -> List[str]:
        """Get list of container IDs for a service"""
        if not self.docker_available:
            return []
        
        try:
            containers = self.docker_client.containers.list(
                filters={'label': f'service={service_name}'}
            )
            return [container.id for container in containers]
        except Exception as e:
            self.logger.error(f"Error getting containers for {service_name}: {e}")
            return []
    
    async def _start_container(self, service_config: ServiceConfig, container_name: str):
        """Start a new container instance"""
        try:
            # Calculate port mapping
            base_port = service_config.port
            existing_containers = len(self._get_service_containers(service_config.name))
            host_port = base_port + existing_containers
            
            container = self.docker_client.containers.run(
                image=service_config.image,
                name=container_name,
                ports={f'{service_config.port}/tcp': host_port},
                environment=service_config.environment,
                volumes=service_config.volumes,
                networks=service_config.networks,
                labels={'service': service_config.name},
                detach=True,
                restart_policy={'Name': 'unless-stopped'}
            )
            
            self.managed_containers[service_config.name].append(container.id)
            self.logger.info(f"Started container {container_name} on port {host_port}")
            
        except Exception as e:
            self.logger.error(f"Error starting container {container_name}: {e}")
            raise
    
    async def _stop_container(self, container_id: str):
        """Stop and remove a container"""
        try:
            container = self.docker_client.containers.get(container_id)
            container.stop(timeout=30)
            container.remove()
            
            # Remove from managed containers
            for service_name, container_list in self.managed_containers.items():
                if container_id in container_list:
                    container_list.remove(container_id)
                    break
            
            self.logger.info(f"Stopped and removed container {container_id}")
            
        except Exception as e:
            self.logger.error(f"Error stopping container {container_id}: {e}")
            raise
    
    def get_service_status(self, service_name: str) -> Dict[str, Any]:
        """Get status of a service"""
        containers = self._get_service_containers(service_name)
        
        status = {
            'service_name': service_name,
            'instance_count': len(containers),
            'containers': []
        }
        
        for container_id in containers:
            try:
                container = self.docker_client.containers.get(container_id)
                status['containers'].append({
                    'id': container_id[:12],
                    'name': container.name,
                    'status': container.status,
                    'created': container.attrs['Created'],
                    'ports': container.ports
                })
            except Exception as e:
                self.logger.error(f"Error getting container status {container_id}: {e}")
        
        return status


class AutoScaler:
    """Main auto-scaling orchestrator"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.resource_monitor = ResourceMonitor(
            collection_interval=config.get('monitoring_interval', 30)
        )
        self.container_manager = ContainerManager()
        
        # Scaling configuration
        self.scaling_rules: Dict[str, ScalingRule] = {}
        self.service_configs: Dict[str, ServiceConfig] = {}
        self.scaling_history: deque = deque(maxlen=1000)
        self.last_scaling_action: Dict[str, datetime] = {}
        
        # Control flags
        self.running = False
        self.scaling_task = None
        
        # Load configuration
        self._load_scaling_rules()
        self._load_service_configs()
    
    def _load_scaling_rules(self):
        """Load scaling rules from configuration"""
        rules_config = self.config.get('scaling_rules', [])
        
        for rule_config in rules_config:
            rule = ScalingRule(
                name=rule_config['name'],
                resource_type=ResourceType(rule_config['resource_type']),
                threshold_up=rule_config['threshold_up'],
                threshold_down=rule_config['threshold_down'],
                duration_minutes=rule_config.get('duration_minutes', 5),
                cooldown_minutes=rule_config.get('cooldown_minutes', 10),
                min_instances=rule_config.get('min_instances', 1),
                max_instances=rule_config.get('max_instances', 10),
                scale_up_count=rule_config.get('scale_up_count', 1),
                scale_down_count=rule_config.get('scale_down_count', 1),
                enabled=rule_config.get('enabled', True)
            )
            self.scaling_rules[rule.name] = rule
    
    def _load_service_configs(self):
        """Load service configurations"""
        services_config = self.config.get('services', [])
        
        for service_config in services_config:
            service = ServiceConfig(
                name=service_config['name'],
                image=service_config['image'],
                port=service_config['port'],
                environment=service_config.get('environment', {}),
                volumes=service_config.get('volumes', []),
                networks=service_config.get('networks', []),
                health_check_path=service_config.get('health_check_path', '/health'),
                min_instances=service_config.get('min_instances', 1),
                max_instances=service_config.get('max_instances', 5)
            )
            self.service_configs[service.name] = service
    
    async def start(self):
        """Start the auto-scaler"""
        self.running = True
        await self.resource_monitor.start_monitoring()
        self.scaling_task = asyncio.create_task(self._scaling_loop())
        self.logger.info("Auto-scaler started")
    
    async def stop(self):
        """Stop the auto-scaler"""
        self.running = False
        await self.resource_monitor.stop_monitoring()
        
        if self.scaling_task:
            self.scaling_task.cancel()
            try:
                await self.scaling_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Auto-scaler stopped")
    
    async def _scaling_loop(self):
        """Main scaling decision loop"""
        while self.running:
            try:
                await self._evaluate_scaling_rules()
                await asyncio.sleep(60)  # Check every minute
            except Exception as e:
                self.logger.error(f"Error in scaling loop: {e}")
                await asyncio.sleep(60)
    
    async def _evaluate_scaling_rules(self):
        """Evaluate all scaling rules and make scaling decisions"""
        current_metrics = self.resource_monitor.get_average_metrics(5)
        
        if not current_metrics:
            return
        
        for rule_name, rule in self.scaling_rules.items():
            if not rule.enabled:
                continue
            
            try:
                await self._evaluate_rule(rule, current_metrics)
            except Exception as e:
                self.logger.error(f"Error evaluating rule {rule_name}: {e}")
    
    async def _evaluate_rule(self, rule: ScalingRule, metrics: Dict[str, float]):
        """Evaluate a single scaling rule"""
        # Get current resource value
        resource_value = self._get_resource_value(rule.resource_type, metrics)
        
        if resource_value is None:
            return
        
        # Check cooldown period
        if not self._is_cooldown_expired(rule.name, rule.cooldown_minutes):
            return
        
        # Determine scaling direction
        scaling_direction = self._determine_scaling_direction(rule, resource_value)
        
        if scaling_direction == ScalingDirection.NONE:
            return
        
        # Find services that match this rule
        matching_services = self._find_matching_services(rule)
        
        for service_name in matching_services:
            await self._execute_scaling_action(rule, service_name, scaling_direction, resource_value)
    
    def _get_resource_value(self, resource_type: ResourceType, metrics: Dict[str, float]) -> Optional[float]:
        """Get the current value for a resource type"""
        mapping = {
            ResourceType.CPU: 'cpu_percent',
            ResourceType.MEMORY: 'memory_percent',
            ResourceType.DISK: 'disk_percent',
            ResourceType.CONNECTIONS: 'active_connections',
            ResourceType.NETWORK: 'load_average'  # Using load average as network proxy
        }
        
        return metrics.get(mapping.get(resource_type))
    
    def _is_cooldown_expired(self, rule_name: str, cooldown_minutes: int) -> bool:
        """Check if cooldown period has expired"""
        last_action = self.last_scaling_action.get(rule_name)
        
        if not last_action:
            return True
        
        cooldown_period = timedelta(minutes=cooldown_minutes)
        return datetime.now() - last_action >= cooldown_period
    
    def _determine_scaling_direction(self, rule: ScalingRule, resource_value: float) -> ScalingDirection:
        """Determine if scaling is needed and in which direction"""
        if resource_value >= rule.threshold_up:
            return ScalingDirection.UP
        elif resource_value <= rule.threshold_down:
            return ScalingDirection.DOWN
        else:
            return ScalingDirection.NONE
    
    def _find_matching_services(self, rule: ScalingRule) -> List[str]:
        """Find services that should be scaled based on the rule"""
        # For now, return all configured services
        # In a more sophisticated implementation, you could have rule-to-service mappings
        return list(self.service_configs.keys())
    
    async def _execute_scaling_action(self, rule: ScalingRule, service_name: str, 
                                    direction: ScalingDirection, trigger_value: float):
        """Execute a scaling action"""
        service_config = self.service_configs.get(service_name)
        if not service_config:
            return
        
        # Get current instance count
        current_containers = self.container_manager._get_service_containers(service_name)
        current_instances = len(current_containers)
        
        # Calculate target instances
        if direction == ScalingDirection.UP:
            target_instances = min(
                current_instances + rule.scale_up_count,
                min(rule.max_instances, service_config.max_instances)
            )
        else:  # ScalingDirection.DOWN
            target_instances = max(
                current_instances - rule.scale_down_count,
                max(rule.min_instances, service_config.min_instances)
            )
        
        # Skip if no change needed
        if target_instances == current_instances:
            return
        
        # Execute scaling
        success = await self.container_manager.scale_service(service_config, target_instances)
        
        # Record scaling event
        event = ScalingEvent(
            timestamp=datetime.now(),
            rule_name=rule.name,
            direction=direction,
            from_instances=current_instances,
            to_instances=target_instances,
            trigger_value=trigger_value,
            reason=f"{rule.resource_type.value} {direction.value} threshold triggered",
            success=success
        )
        
        self.scaling_history.append(event)
        self.last_scaling_action[rule.name] = datetime.now()
        
        if success:
            self.logger.info(
                f"Scaled {service_name} {direction.value} from {current_instances} to {target_instances} "
                f"instances (trigger: {rule.resource_type.value}={trigger_value:.2f})"
            )
        else:
            self.logger.error(f"Failed to scale {service_name}")
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current scaling status"""
        status = {
            'running': self.running,
            'rules': {name: rule.__dict__ for name, rule in self.scaling_rules.items()},
            'services': {},
            'recent_events': []
        }
        
        # Get service status
        for service_name in self.service_configs.keys():
            status['services'][service_name] = self.container_manager.get_service_status(service_name)
        
        # Get recent scaling events
        recent_events = list(self.scaling_history)[-10:]
        status['recent_events'] = [
            {
                'timestamp': event.timestamp.isoformat(),
                'rule_name': event.rule_name,
                'direction': event.direction.value,
                'from_instances': event.from_instances,
                'to_instances': event.to_instances,
                'trigger_value': event.trigger_value,
                'reason': event.reason,
                'success': event.success
            }
            for event in recent_events
        ]
        
        return status
    
    def get_resource_metrics(self) -> Dict[str, Any]:
        """Get current resource metrics"""
        current_metrics = self.resource_monitor.get_average_metrics(5)
        recent_metrics = self.resource_monitor.get_recent_metrics(30)
        
        return {
            'current': current_metrics,
            'history_count': len(recent_metrics),
            'collection_interval': self.resource_monitor.collection_interval
        }


# Example configuration
DEFAULT_CONFIG = {
    'monitoring_interval': 30,
    'scaling_rules': [
        {
            'name': 'cpu_scaling',
            'resource_type': 'cpu',
            'threshold_up': 80.0,
            'threshold_down': 30.0,
            'duration_minutes': 5,
            'cooldown_minutes': 10,
            'min_instances': 1,
            'max_instances': 5,
            'scale_up_count': 1,
            'scale_down_count': 1,
            'enabled': True
        },
        {
            'name': 'memory_scaling',
            'resource_type': 'memory',
            'threshold_up': 85.0,
            'threshold_down': 40.0,
            'duration_minutes': 5,
            'cooldown_minutes': 15,
            'min_instances': 1,
            'max_instances': 3,
            'scale_up_count': 1,
            'scale_down_count': 1,
            'enabled': True
        }
    ],
    'services': [
        {
            'name': 'trading-system',
            'image': 'trading-system:latest',
            'port': 8000,
            'environment': {
                'ENV': 'production'
            },
            'min_instances': 1,
            'max_instances': 5
        }
    ]
}


if __name__ == "__main__":
    # Test auto-scaling functionality
    async def test_auto_scaling():
        logging.basicConfig(level=logging.INFO)
        
        # Create auto-scaler with test configuration
        auto_scaler = AutoScaler(DEFAULT_CONFIG)
        
        try:
            # Start auto-scaler
            await auto_scaler.start()
            
            # Run for a short time
            await asyncio.sleep(120)
            
            # Print status
            status = auto_scaler.get_scaling_status()
            print("Scaling status:", json.dumps(status, indent=2, default=str))
            
            metrics = auto_scaler.get_resource_metrics()
            print("Resource metrics:", json.dumps(metrics, indent=2, default=str))
            
        finally:
            await auto_scaler.stop()
    
    asyncio.run(test_auto_scaling())