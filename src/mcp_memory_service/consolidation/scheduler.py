# Copyright 2024 Heinrich Krupp
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""APScheduler integration for autonomous consolidation operations."""

import logging
import os
from typing import Dict, Any, Optional
from datetime import datetime

try:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from apscheduler.triggers.cron import CronTrigger
    from apscheduler.triggers.interval import IntervalTrigger
    from apscheduler.jobstores.memory import MemoryJobStore
    from apscheduler.executors.asyncio import AsyncIOExecutor
    from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR
    APSCHEDULER_AVAILABLE = True
except ImportError:
    APSCHEDULER_AVAILABLE = False

from .consolidator import DreamInspiredConsolidator
from .belief_service import BeliefService

class ConsolidationScheduler:
    """
    Scheduler for autonomous consolidation operations.
    
    Integrates with APScheduler to run consolidation operations at specified intervals
    based on time horizons (daily, weekly, monthly, quarterly, yearly).
    """
    
    def __init__(
        self, 
        consolidator: DreamInspiredConsolidator,
        schedule_config: Dict[str, str],
        enabled: bool = True
    ):
        self.consolidator = consolidator
        self.schedule_config = schedule_config
        self.enabled = enabled
        self.logger = logging.getLogger(__name__)
        
        # Job execution tracking
        self.job_history = []
        self.last_execution_times = {}
        self.execution_stats = {
            'total_jobs': 0,
            'successful_jobs': 0,
            'failed_jobs': 0
        }
        
        # Initialize scheduler if APScheduler is available
        if APSCHEDULER_AVAILABLE and enabled:
            self.scheduler = AsyncIOScheduler(
                jobstores={'default': MemoryJobStore()},
                executors={'default': AsyncIOExecutor()},
                job_defaults={
                    'coalesce': True,  # Combine multiple pending executions
                    'max_instances': 1,  # Only one instance of each job at a time
                    'misfire_grace_time': 3600  # 1 hour grace period for missed jobs
                }
            )
            
            # Add event listeners
            self.scheduler.add_listener(self._job_executed_listener, EVENT_JOB_EXECUTED)
            self.scheduler.add_listener(self._job_error_listener, EVENT_JOB_ERROR)
        else:
            self.scheduler = None
            if not APSCHEDULER_AVAILABLE:
                self.logger.warning("APScheduler not available - consolidation scheduling disabled")
            elif not enabled:
                self.logger.info("Consolidation scheduling disabled by configuration")
    
    async def start(self) -> bool:
        """Start the consolidation scheduler."""
        if not self.scheduler:
            return False
        
        try:
            # Add consolidation jobs based on configuration
            self._schedule_consolidation_jobs()
            
            # Start the scheduler
            self.scheduler.start()
            self.logger.info("Consolidation scheduler started successfully")
            
            # Log scheduled jobs
            jobs = self.scheduler.get_jobs()
            for job in jobs:
                self.logger.info(f"Scheduled job: {job.id} - next run: {job.next_run_time}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start consolidation scheduler: {e}")
            return False
    
    async def stop(self) -> bool:
        """Stop the consolidation scheduler."""
        if not self.scheduler:
            return True
        
        try:
            self.scheduler.shutdown(wait=True)
            self.logger.info("Consolidation scheduler stopped")
            return True
        except Exception as e:
            self.logger.error(f"Error stopping consolidation scheduler: {e}")
            return False
    
    def _schedule_consolidation_jobs(self):
        """Schedule consolidation jobs based on configuration."""
        time_horizons = ['daily', 'weekly', 'monthly', 'quarterly', 'yearly']
        
        for horizon in time_horizons:
            schedule_spec = self.schedule_config.get(horizon, 'disabled')
            
            if schedule_spec == 'disabled':
                self.logger.debug(f"Consolidation for {horizon} horizon is disabled")
                continue
            
            try:
                trigger = self._create_trigger(horizon, schedule_spec)
                if trigger:
                    job_id = f"consolidation_{horizon}"
                    self.scheduler.add_job(
                        func=self._run_consolidation_job,
                        trigger=trigger,
                        args=[horizon],
                        id=job_id,
                        name=f"Consolidation - {horizon.title()}",
                        replace_existing=True
                    )
                    self.logger.info(f"Scheduled {horizon} consolidation: {schedule_spec}")
                
            except Exception as e:
                self.logger.error(f"Error scheduling {horizon} consolidation: {e}")
    
    def _create_trigger(self, horizon: str, schedule_spec: str):
        """Create APScheduler trigger from schedule specification."""
        try:
            if horizon == 'daily':
                # Daily format: "HH:MM" (e.g., "02:00")
                hour, minute = map(int, schedule_spec.split(':'))
                return CronTrigger(hour=hour, minute=minute)
            
            elif horizon == 'weekly':
                # Weekly format: "DAY HH:MM" (e.g., "SUN 03:00")
                day_time = schedule_spec.split(' ')
                if len(day_time) != 2:
                    raise ValueError(f"Invalid weekly schedule format: {schedule_spec}")
                
                day_map = {
                    'MON': 0, 'TUE': 1, 'WED': 2, 'THU': 3, 
                    'FRI': 4, 'SAT': 5, 'SUN': 6
                }
                
                day = day_map.get(day_time[0].upper())
                if day is None:
                    raise ValueError(f"Invalid day: {day_time[0]}")
                
                hour, minute = map(int, day_time[1].split(':'))
                return CronTrigger(day_of_week=day, hour=hour, minute=minute)
            
            elif horizon == 'monthly':
                # Monthly format: "DD HH:MM" (e.g., "01 04:00")
                day_time = schedule_spec.split(' ')
                if len(day_time) != 2:
                    raise ValueError(f"Invalid monthly schedule format: {schedule_spec}")
                
                day = int(day_time[0])
                hour, minute = map(int, day_time[1].split(':'))
                return CronTrigger(day=day, hour=hour, minute=minute)
            
            elif horizon == 'quarterly':
                # Quarterly format: "MM-DD HH:MM" (e.g., "01-01 05:00")
                # Run on the first day of quarters (Jan, Apr, Jul, Oct)
                parts = schedule_spec.split(' ')
                if len(parts) != 2:
                    raise ValueError(f"Invalid quarterly schedule format: {schedule_spec}")
                
                month_day = parts[0].split('-')
                if len(month_day) != 2:
                    raise ValueError(f"Invalid quarterly date format: {parts[0]}")
                
                day = int(month_day[1])
                hour, minute = map(int, parts[1].split(':'))
                
                # Quarters: Jan(1), Apr(4), Jul(7), Oct(10)
                return CronTrigger(month='1,4,7,10', day=day, hour=hour, minute=minute)
            
            elif horizon == 'yearly':
                # Yearly format: "MM-DD HH:MM" (e.g., "01-01 06:00")
                parts = schedule_spec.split(' ')
                if len(parts) != 2:
                    raise ValueError(f"Invalid yearly schedule format: {schedule_spec}")
                
                month_day = parts[0].split('-')
                if len(month_day) != 2:
                    raise ValueError(f"Invalid yearly date format: {parts[0]}")
                
                month = int(month_day[0])
                day = int(month_day[1])
                hour, minute = map(int, parts[1].split(':'))
                
                return CronTrigger(month=month, day=day, hour=hour, minute=minute)
            
            else:
                self.logger.error(f"Unknown time horizon: {horizon}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error creating trigger for {horizon} with spec '{schedule_spec}': {e}")
            return None
    
    async def _run_consolidation_job(self, time_horizon: str):
        """Execute a consolidation job for the specified time horizon."""
        job_start_time = datetime.now()
        self.logger.info(f"Starting scheduled {time_horizon} consolidation")
        
        try:
            # Run the consolidation
            report = await self.consolidator.consolidate(time_horizon)
            
            # Run belief derivation (opt-in via MCP_BELIEFS_ENABLED)
            belief_stats = {}
            beliefs_enabled = os.getenv("MCP_BELIEFS_ENABLED", "false").lower() in ("true", "1", "yes")
            if beliefs_enabled and hasattr(self.consolidator, 'storage'):
                try:
                    belief_svc = BeliefService(self.consolidator.storage)
                    belief_stats = await belief_svc.derive_beliefs()
                except Exception as be:
                    self.logger.warning(f"Belief derivation error (non-fatal): {be}")

            
            # Record successful execution
            self.execution_stats['successful_jobs'] += 1
            self.last_execution_times[time_horizon] = job_start_time
            
            # Add to job history
            job_record = {
                'time_horizon': time_horizon,
                'start_time': job_start_time,
                'end_time': datetime.now(),
                'status': 'success',
                'memories_processed': report.memories_processed,
                'associations_discovered': report.associations_discovered,
                'clusters_created': report.clusters_created,
                'memories_compressed': report.memories_compressed,
                'memories_archived': report.memories_archived,
                'beliefs': belief_stats,
                'errors': report.errors
            }
            
            self._add_job_to_history(job_record)
            
            # Log success
            duration = (job_record['end_time'] - job_record['start_time']).total_seconds()
            self.logger.info(
                f"Completed {time_horizon} consolidation successfully in {duration:.2f}s: "
                f"{report.memories_processed} memories processed, "
                f"{report.associations_discovered} associations, "
                f"{report.clusters_created} clusters, "
                f"{report.memories_compressed} compressed, "
                f"{report.memories_archived} archived"
            )
            
        except Exception as e:
            # Record failed execution
            self.execution_stats['failed_jobs'] += 1
            
            job_record = {
                'time_horizon': time_horizon,
                'start_time': job_start_time,
                'end_time': datetime.now(),
                'status': 'failed',
                'error': str(e),
                'memories_processed': 0,
                'associations_discovered': 0,
                'clusters_created': 0,
                'memories_compressed': 0,
                'memories_archived': 0,
                'errors': [str(e)]
            }
            
            self._add_job_to_history(job_record)
            
            self.logger.error(f"Failed {time_horizon} consolidation: {e}")
            raise
    
    def _add_job_to_history(self, job_record: Dict[str, Any]):
        """Add job record to history with size limit."""
        self.job_history.append(job_record)
        
        # Keep only last 100 job records
        if len(self.job_history) > 100:
            self.job_history = self.job_history[-100:]
    
    def _job_executed_listener(self, event):
        """Handle job execution events."""
        self.execution_stats['total_jobs'] += 1
        self.logger.debug(f"Job executed: {event.job_id}")
    
    def _job_error_listener(self, event):
        """Handle job error events."""
        self.logger.error(f"Job error: {event.job_id} - {event.exception}")
    
    async def trigger_consolidation(self, time_horizon: str, immediate: bool = True) -> bool:
        """Manually trigger a consolidation job."""
        if not self.scheduler:
            self.logger.error("Scheduler not available")
            return False
        
        try:
            if immediate:
                # Run immediately
                await self._run_consolidation_job(time_horizon)
                return True
            else:
                # Schedule to run in 1 minute
                job_id = f"manual_consolidation_{time_horizon}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                trigger = IntervalTrigger(seconds=60)  # Run once after 60 seconds
                
                self.scheduler.add_job(
                    func=self._run_consolidation_job,
                    trigger=trigger,
                    args=[time_horizon],
                    id=job_id,
                    name=f"Manual Consolidation - {time_horizon.title()}",
                    max_instances=1
                )
                
                self.logger.info(f"Scheduled manual {time_horizon} consolidation")
                return True
                
        except Exception as e:
            self.logger.error(f"Error triggering {time_horizon} consolidation: {e}")
            return False
    
    async def get_scheduler_status(self) -> Dict[str, Any]:
        """Get scheduler status and job information."""
        if not self.scheduler:
            return {
                'enabled': False,
                'reason': 'APScheduler not available or disabled'
            }
        
        jobs = self.scheduler.get_jobs()
        job_info = []
        
        for job in jobs:
            job_info.append({
                'id': job.id,
                'name': job.name,
                'next_run_time': job.next_run_time.isoformat() if job.next_run_time else None,
                'trigger': str(job.trigger)
            })
        
        return {
            'enabled': True,
            'running': self.scheduler.running,
            'jobs': job_info,
            'execution_stats': self.execution_stats.copy(),
            'last_execution_times': {
                horizon: time.isoformat() for horizon, time in self.last_execution_times.items()
            },
            'recent_jobs': self.job_history[-10:]  # Last 10 jobs
        }
    
    async def update_schedule(self, new_schedule_config: Dict[str, str]) -> bool:
        """Update the consolidation schedule."""
        if not self.scheduler:
            return False
        
        try:
            # Remove existing consolidation jobs
            job_ids = [f"consolidation_{horizon}" for horizon in ['daily', 'weekly', 'monthly', 'quarterly', 'yearly']]
            
            for job_id in job_ids:
                if self.scheduler.get_job(job_id):
                    self.scheduler.remove_job(job_id)
            
            # Update configuration
            self.schedule_config = new_schedule_config
            
            # Re-schedule jobs
            self._schedule_consolidation_jobs()
            
            self.logger.info("Consolidation schedule updated successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating consolidation schedule: {e}")
            return False
    
    async def pause_consolidation(self, time_horizon: Optional[str] = None) -> bool:
        """Pause consolidation jobs (all or specific horizon)."""
        if not self.scheduler:
            return False
        
        try:
            if time_horizon:
                job_id = f"consolidation_{time_horizon}"
                job = self.scheduler.get_job(job_id)
                if job:
                    self.scheduler.pause_job(job_id)
                    self.logger.info(f"Paused {time_horizon} consolidation")
                else:
                    self.logger.warning(f"No job found for {time_horizon} consolidation")
            else:
                # Pause all consolidation jobs
                jobs = self.scheduler.get_jobs()
                for job in jobs:
                    if job.id.startswith('consolidation_'):
                        self.scheduler.pause_job(job.id)
                
                self.logger.info("Paused all consolidation jobs")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error pausing consolidation: {e}")
            return False
    
    async def resume_consolidation(self, time_horizon: Optional[str] = None) -> bool:
        """Resume consolidation jobs (all or specific horizon)."""
        if not self.scheduler:
            return False
        
        try:
            if time_horizon:
                job_id = f"consolidation_{time_horizon}"
                job = self.scheduler.get_job(job_id)
                if job:
                    self.scheduler.resume_job(job_id)
                    self.logger.info(f"Resumed {time_horizon} consolidation")
                else:
                    self.logger.warning(f"No job found for {time_horizon} consolidation")
            else:
                # Resume all consolidation jobs
                jobs = self.scheduler.get_jobs()
                for job in jobs:
                    if job.id.startswith('consolidation_'):
                        self.scheduler.resume_job(job.id)
                
                self.logger.info("Resumed all consolidation jobs")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error resuming consolidation: {e}")
            return False