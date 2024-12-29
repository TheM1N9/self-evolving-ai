from agent import ScreenAgent, VerifierAgent
import asyncio
import google.generativeai as genai
import time
import numpy as np
import cv2
from io import BytesIO
import json
import os
from datetime import datetime


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
        """Add new entry to chat history and save"""
        entry["timestamp"] = datetime.now().isoformat()
        self.chat_history.append(entry)
        self.save_chat_history()
        # Update chat history for all agents
        for agent in self.agents.values():
            agent.chat_history = self.chat_history

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
        max_verification_attempts = 3  # Maximum number of verification attempts
        max_retries = 10  # Maximum retries for file activation

        while self.running:
            try:
                # Record initial state
                video_path = self.agents["goal_generator"].record_screen()
                print(f"Video recorded: {video_path}")
                execution_video_path = None  # Initialize to None

                try:
                    with open(video_path, "rb") as f:
                        video_file = genai.upload_file(f, mime_type="video/avi")
                        print(f"Video file uploaded: {video_file.name}")

                    # Get current screen snapshot
                    screen = self.agents["goal_generator"].screen_capture.grab(
                        self.agents["goal_generator"].screen_capture.monitors[1]
                    )
                    screen_array = np.array(screen)
                    screen_image = genai.upload_file(
                        BytesIO(cv2.imencode(".png", screen_array)[1]),
                        mime_type="image/png",
                    )

                    # Wait for file to become ACTIVE
                    retry_count = 0
                    while (
                        video_file.state.name == "PROCESSING"
                        and retry_count < max_retries
                    ):
                        time.sleep(1)
                        retry_count += 1
                        video_file = genai.get_file(video_file.name)

                    if video_file.state.name != "ACTIVE":
                        raise Exception(
                            f"File failed to become active. State: {video_file.state.name}"
                        )

                    # First clean up old videos
                    await self.cleanup_old_videos()
                    # Then add new video to history
                    self.video_history.append(video_file)

                    # Generate goal with shared chat history
                    goal = await self.agents["goal_generator"].generate_goal(
                        video_file=video_file,
                        video_history=self.video_history,
                        current_screen=screen_image,
                    )

                    if goal:
                        # Add goal generation to chat history
                        self.append_chat_history(
                            {
                                "type": "goal_generation",
                                "goal": goal,
                                "agent": "goal_generator",
                            }
                        )

                        commands = self.agents["goal_executor"].goal_parser(goal)
                        verification_attempts = 0
                        execution_success = False

                        while (
                            not execution_success
                            and verification_attempts < max_verification_attempts
                        ):
                            verification_attempts += 1
                            print(
                                f"Execution attempt {verification_attempts}/{max_verification_attempts}"
                            )

                            # Start recording before execution
                            execution_video_path = self.agents[
                                "goal_generator"
                            ].start_recording()

                            # Execute commands
                            execution_result = await self.agents[
                                "goal_executor"
                            ].execute_goal(goal, commands)
                            await self.agents["goal_executor"].wait_for_completion()

                            # Stop recording after execution
                            self.agents["goal_generator"].stop_recording()

                            if execution_result.get("success"):
                                # Upload the execution video
                                with open(execution_video_path, "rb") as f:
                                    execution_video = genai.upload_file(
                                        f, mime_type="video/avi"
                                    )

                                # Wait for execution video to become ACTIVE
                                retry_count = 0
                                while (
                                    execution_video.state.name == "PROCESSING"
                                    and retry_count < max_retries
                                ):
                                    time.sleep(1)
                                    retry_count += 1
                                    execution_video = genai.get_file(
                                        execution_video.name
                                    )

                                # Get current screen elements
                                current_elements = self.agents[
                                    "goal_executor"
                                ].get_clickable_elements()

                                # Verify execution with video and elements
                                verification = await self.agents[
                                    "verifier"
                                ].verify_execution(
                                    goal=goal,
                                    original_commands=commands,
                                    execution_video=execution_video,
                                    execution_logs=execution_result.get("logs", []),
                                    available_elements=current_elements,
                                )

                                if verification.get("execution") == "success":
                                    print("Goal executed successfully!")
                                    execution_success = True

                                    # Add verification to chat history
                                    self.append_chat_history(
                                        {
                                            "type": "verification",
                                            "goal": goal,
                                            "success": True,
                                            "agent": "verifier",
                                        }
                                    )
                                else:
                                    print(
                                        f"Execution attempt {verification_attempts} failed, trying new commands..."
                                    )
                                    commands = verification.get("commands")

                                    # Add failed verification to chat history
                                    self.append_chat_history(
                                        {
                                            "type": "verification",
                                            "goal": goal,
                                            "success": False,
                                            "reason": verification.get("reason"),
                                            "new_commands": commands,
                                            "agent": "verifier",
                                        }
                                    )
                            else:
                                # Add failed execution to chat history
                                self.append_chat_history(
                                    {
                                        "type": "execution",
                                        "goal": goal,
                                        "commands": commands,
                                        "success": False,
                                        "error": execution_result.get("error"),
                                        "agent": "goal_executor",
                                    }
                                )

                        if not execution_success:
                            print(
                                f"Failed to execute goal after {max_verification_attempts} attempts"
                            )

                finally:
                    # Clean up local video files
                    self.agents["goal_generator"].cleanup_video(video_path)
                    if execution_video_path:  # Only cleanup if path exists
                        self.agents["goal_generator"].cleanup_video(
                            execution_video_path
                        )

                await asyncio.sleep(self.recording_interval)

            except Exception as e:
                print(f"Error in agent loop: {str(e)}")
                await asyncio.sleep(self.error_retry_delay)

    def stop(self):
        """Stop the orchestrator loop and cleanup"""
        self.running = False
        # Additional cleanup could be added here
