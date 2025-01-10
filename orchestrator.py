from agent import ScreenAgent, VerifierAgent, GoalSimplifierAgent, LearningAgent
import asyncio
import google.generativeai as genai
import time
import numpy as np
import cv2
from io import BytesIO
import json
import os
from datetime import datetime
import re


class AgentOrchestrator:
    def __init__(self, api_key):
        self.agents = {}
        self.api_key = api_key
        self.running = False
        self.error_retry_delay = 5  # Configurable retry delay
        self.recording_interval = 1  # Configurable interval between recordings
        self.video_history = []  # Store last 3 video files
        self.max_history = 3  # Maximum number of videos to keep

        # Centralized chat history
        self.chat_history = []
        self.chat_history_file = "agent_history/chat_history.json"
        os.makedirs("agent_history", exist_ok=True)
        self.load_chat_history()

        # Create goal simplifier agent
        self.create_agent("goal_simplifier", agent_type="simplifier", is_async=False)

        # Create learning agent
        self.create_agent("learner", agent_type="learner", is_async=False)

    def load_chat_history(self):
        """Load chat history from file at startup"""
        try:
            if os.path.exists(self.chat_history_file):
                with open(self.chat_history_file, "r") as f:
                    self.chat_history = json.load(f)
            else:
                self.chat_history = []
        except Exception as e:
            print(f"Failed to load chat history: {e}")
            self.chat_history = []

    def save_chat_history(self):
        """Save chat history to file"""
        try:
            with open(self.chat_history_file, "w") as f:
                json.dump(self.chat_history, f, indent=2)
        except Exception as e:
            print(f"Failed to save chat history: {e}")

    def append_chat_history(self, entry):
        """Append entry to chat history with error recovery"""
        try:
            # Load existing history with recovery
            history = []
            if os.path.exists("agent_history/chat_history.json"):
                try:
                    with open("agent_history/chat_history.json", "r") as f:
                        history = json.load(f)
                except json.JSONDecodeError:
                    # Backup corrupted file
                    backup_path = (
                        f"agent_history/chat_history_backup_{int(time.time())}.json"
                    )
                    os.rename("agent_history/chat_history.json", backup_path)
                    print(f"Corrupted history backed up to: {backup_path}")
                    history = []  # Start fresh

            # Sanitize the entry
            sanitized_entry = {}
            for key, value in entry.items():
                if isinstance(value, (str, int, bool, list, dict)):
                    sanitized_entry[key] = value
                else:
                    # Convert non-standard types to string representation
                    sanitized_entry[key] = str(value)

            # Add timestamp
            sanitized_entry["timestamp"] = datetime.now().isoformat()

            # Append new entry
            history.append(sanitized_entry)

            # Save with pretty printing
            with open("agent_history/chat_history.json", "w") as f:
                json.dump(history, f, indent=4, ensure_ascii=False)

        except Exception as e:
            print(f"Failed to save chat history: {str(e)}")
            # Create new history file if needed
            if not os.path.exists("agent_history/chat_history.json"):
                with open("agent_history/chat_history.json", "w") as f:
                    json.dump([], f)

    def create_agent(self, agent_id, agent_type="screen", is_async=False):
        """Create a new agent with specific role and shared chat history"""
        if agent_type == "screen":
            self.agents[agent_id] = ScreenAgent(
                self.api_key, is_async=is_async, role=agent_id
            )
        elif agent_type == "verifier":
            self.agents[agent_id] = VerifierAgent(
                self.api_key, is_async=is_async, role=agent_id
            )
        elif agent_type == "simplifier":
            self.agents[agent_id] = GoalSimplifierAgent(
                self.api_key, is_async=is_async, role=agent_id
            )
        elif agent_type == "learner":
            self.agents[agent_id] = LearningAgent(
                self.api_key, is_async=is_async, role=agent_id
            )

        # Share chat history with new agent
        self.agents[agent_id].chat_history = self.chat_history
        return self.agents[agent_id]

    async def cleanup_old_videos(self):
        """Delete old video files from the API"""
        try:
            while (
                len(self.video_history) > self.max_history - 1
            ):  # Leave room for new video
                old_video = self.video_history.pop(0)  # Remove oldest video
                try:
                    # Delete the file from Gemini API
                    genai.delete_file(old_video.name)
                    print(f"Deleted old video file: {old_video.name}")
                except Exception as e:
                    print(f"Error deleting video file {old_video.name}: {str(e)}")

        except Exception as e:
            print(f"Error during video cleanup: {str(e)}")

    async def run_agent_loop(self):
        """Coordinate agents in a loop with improved error handling and cleanup"""
        self.running = True
        max_verification_attempts = 3
        max_retries = 10
        self.continuous_video_history = []  # For goal generator's context

        while self.running:
            try:
                # Start continuous recording for goal generator
                continuous_video_path = self.agents["goal_generator"].start_recording()
                print(f"Started continuous recording for goal generator")

                try:
                    # Get current screen snapshot
                    screen = self.agents["goal_generator"].screen_capture.grab(
                        self.agents["goal_generator"].screen_capture.monitors[1]
                    )
                    screen_array = np.array(screen)
                    screen_image = genai.upload_file(
                        BytesIO(cv2.imencode(".png", screen_array)[1]),
                        mime_type="image/png",
                    )

                    # Get last three continuous videos for context
                    recent_videos = (
                        self.continuous_video_history[-3:]
                        if len(self.continuous_video_history) >= 3
                        else self.continuous_video_history
                    )
                    latest_video = (
                        self.continuous_video_history[-1]
                        if self.continuous_video_history
                        else None
                    )

                    # Process experiences before goal generation
                    learning_result = await self.agents["learner"].process_experience(
                        video_history=self.continuous_video_history,
                        current_screen=screen_image,
                        chat_history=self.chat_history,
                    )

                    if learning_result["success"]:
                        self.append_chat_history(
                            {
                                "type": "learning_update",
                                "updates": learning_result["updates"],
                                "agent": "learner",
                            }
                        )

                    # Generate main goal
                    goal = await self.agents["goal_generator"].generate_goal(
                        video_file=latest_video,
                        video_history=recent_videos,
                        current_screen=screen_image,
                    )

                    if goal:
                        # Initialize sub-goals tracking
                        previous_sub_goals = []
                        main_goal_completed = False

                        while not main_goal_completed:
                            # Get current screen elements for context
                            current_elements = self.agents[
                                "goal_executor"
                            ].get_clickable_elements()

                            # Get current screen snapshot
                            screen = self.agents["goal_simplifier"].screen_capture.grab(
                                self.agents["goal_simplifier"].screen_capture.monitors[
                                    1
                                ]
                            )
                            screen_array = np.array(screen)
                            screen_image = genai.upload_file(
                                BytesIO(cv2.imencode(".png", screen_array)[1]),
                                mime_type="image/png",
                            )

                            # Get next batch of sub-goals
                            simplified_goal_data = await self.agents[
                                "goal_simplifier"
                            ].simplify_goal(
                                goal["goal"],
                                current_screen=screen_image,
                                available_elements=current_elements,
                                previous_goals=previous_sub_goals,
                            )

                            if not simplified_goal_data:
                                break

                            # Check if main goal is completed
                            main_goal_completed = simplified_goal_data.get(
                                "main_goal_completed", False
                            )
                            if main_goal_completed:
                                break

                            # Execute each sub-goal with dependency handling
                            for sub_goal in simplified_goal_data["sub_goals"]:
                                # Check prerequisites
                                prerequisites_met = True
                                for prereq in sub_goal.get("prerequisites", []):
                                    if prereq not in [
                                        g["goal"]
                                        for g in previous_sub_goals
                                        if g["status"] == "completed"
                                    ]:
                                        prerequisites_met = False
                                        break

                                if not prerequisites_met:
                                    print(
                                        f"Prerequisites not met for sub-goal: {sub_goal['goal']}"
                                    )
                                    continue

                                # Check dependencies
                                dependencies_met = True
                                for dep_id in sub_goal.get("depends_on", []):
                                    if dep_id not in [
                                        g.get("id")
                                        for g in previous_sub_goals
                                        if g["status"] == "completed"
                                    ]:
                                        dependencies_met = False
                                        break

                                if not dependencies_met:
                                    print(
                                        f"Dependencies not met for sub-goal: {sub_goal['goal']}"
                                    )
                                    continue

                                execution_video_path = self.agents[
                                    "goal_executor"
                                ].start_recording()

                                try:
                                    # Try primary commands first
                                    execution_result = await self.agents[
                                        "goal_executor"
                                    ].execute_goal({"goal": sub_goal["goal"]})

                                    # If primary commands fail, try fallback commands
                                    if not execution_result.get(
                                        "success"
                                    ) and sub_goal.get("fallback_commands"):
                                        print(
                                            f"Primary commands failed, trying fallback for: {sub_goal['goal']}"
                                        )
                                        execution_result = await self.agents[
                                            "goal_executor"
                                        ].execute_goal(
                                            {"goal": sub_goal["goal"]},
                                            sub_goal["fallback_commands"],
                                        )

                                    # Record the result
                                    sub_goal["status"] = (
                                        "completed"
                                        if execution_result.get("success")
                                        else "failed"
                                    )
                                    sub_goal["execution_result"] = execution_result
                                    previous_sub_goals.append(sub_goal)

                                    # Log sub-goal execution
                                    self.append_chat_history(
                                        {
                                            "type": "sub_goal_execution",
                                            "sub_goal": sub_goal,
                                            "result": execution_result,
                                            "progress_percentage": simplified_goal_data.get(
                                                "progress_percentage", 0
                                            ),
                                            "agent": "goal_executor",
                                        }
                                    )

                                    # If execution failed, get new sub-goals immediately
                                    if not execution_result.get("success"):
                                        print(
                                            f"Sub-goal failed: {sub_goal['goal']}, updating plan..."
                                        )
                                        break

                                finally:
                                    self.agents["goal_executor"].stop_recording()
                                    if execution_video_path:
                                        self.agents["goal_executor"].cleanup_video(
                                            execution_video_path
                                        )

                                await asyncio.sleep(1)  # Small delay between sub-goals

                            # If main goal is completed, process the learning
                            if main_goal_completed:
                                learning_result = await self.agents[
                                    "learner"
                                ].process_experience(
                                    video_history=self.continuous_video_history,
                                    current_screen=screen_image,
                                    chat_history=self.chat_history,
                                )

                                self.append_chat_history(
                                    {
                                        "type": "final_learning",
                                        "goal": goal,
                                        "sub_goals": previous_sub_goals,
                                        "learning": learning_result,
                                        "agent": "learner",
                                    }
                                )

                    # Stop and process continuous recording
                    self.agents["goal_generator"].stop_recording()
                    try:
                        with open(continuous_video_path, "rb") as f:
                            continuous_video = genai.upload_file(
                                f, mime_type="video/avi"
                            )

                        # Wait for continuous video to become ACTIVE
                        retry_count = 0
                        while (
                            continuous_video.state.name == "PROCESSING"
                            and retry_count < max_retries
                        ):
                            time.sleep(1)
                            retry_count += 1
                            continuous_video = genai.get_file(continuous_video.name)

                        # Update continuous video history
                        self.continuous_video_history.append(continuous_video)
                        while len(self.continuous_video_history) > self.max_history:
                            old_video = self.continuous_video_history.pop(0)
                            try:
                                genai.delete_file(old_video.name)
                            except Exception as e:
                                print(f"Failed to delete old video: {e}")

                    except Exception as e:
                        print(f"Failed to process continuous recording: {e}")

                finally:
                    # Cleanup continuous recording with error handling
                    if continuous_video_path:
                        try:
                            self.agents["goal_generator"].cleanup_video(
                                continuous_video_path
                            )
                        except Exception as e:
                            print(f"Failed to cleanup continuous recording: {e}")
                            # Add delay to allow file to be released
                            await asyncio.sleep(1)
                            try:
                                self.agents["goal_generator"].cleanup_video(
                                    continuous_video_path
                                )
                            except Exception as e:
                                print(f"Second cleanup attempt failed: {e}")

                await asyncio.sleep(self.recording_interval)

            except Exception as e:
                print(f"Error in agent loop: {str(e)}")
                await asyncio.sleep(self.error_retry_delay)

    def stop(self):
        """Stop the orchestrator loop and cleanup"""
        self.running = False
        # Additional cleanup could be added here
