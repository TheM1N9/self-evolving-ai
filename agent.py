import cv2
import mss
import google.generativeai as genai
import numpy as np
import os
from datetime import datetime
import tempfile
import time
import logging
import json
import asyncio
import pyautogui
from io import BytesIO
from PIL import Image
import pytesseract
import re

# pyautogui.FAILSAFE = True  # Safety feature: move mouse to corner to abort


class ScreenAgent:
    def __init__(self, api_key, is_async=False, role=None):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.0-flash-exp")
        self.screen_capture = mss.mss()
        self.memory = []
        self.is_async = is_async
        self.running = False
        self.async_task = None
        self.clickable_elements = []  # Store clickable elements

        # Video recording settings
        self.fps = 10  # Frames per second of the recorded video
        self.recording_duration = 5  # seconds
        self.temp_dir = tempfile.gettempdir()

        # Setup logging
        self.setup_logging()

        # Setup output directory
        self.output_dir = "agent_output"
        os.makedirs(self.output_dir, exist_ok=True)

        self.state = {
            "current_goal": None,
            "completed_tasks": [],
            "failed_tasks": [],
            "reflections": [],
            "next_goals": [],
        }

        self.input_controller = pyautogui
        self.role = role

        # self.chat = None
        # if role == "goal_generator":
        #     self.chat = self.model.start_chat(history=[])

        self.chat_history = []

        # Add safety boundaries and movement settings
        self.screen_bounds = {
            "left": 0,
            "top": 0,
            "right": pyautogui.size().width,
            "bottom": pyautogui.size().height,
        }
        self.movement_duration = 0.5  # Duration for mouse movements
        pyautogui.PAUSE = 0.1  # Add small pause between PyAutoGUI commands
        pyautogui.FAILSAFE = True  # Enable fail-safe corner

        # Initialize CV settings
        self.template_dir = "ui_templates"
        os.makedirs(self.template_dir, exist_ok=True)
        self.confidence_threshold = 0.8
        self.clickable_elements_cache = {}

        # Set Tesseract path
        pytesseract.pytesseract.tesseract_cmd = (
            r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Adjust path if needed
        )

        # Enhanced vision settings
        self.ocr_confidence_threshold = 0.7
        self.clickable_elements = []  # Store clickable elements

        # Initialize OCR
        pytesseract.pytesseract.tesseract_cmd = (
            r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        )

        # Cache for element positions
        self.element_cache = {}
        self.cache_timeout = 1.0  # Cache timeout in seconds
        self.last_cache_time = 0

        self.pending_commands = []
        self.command_lock = asyncio.Lock()

        # Setup goal history
        self.goals_file = "agent_history/goals.json"
        os.makedirs("agent_history", exist_ok=True)
        self.load_goal_history()

        # Add video recording state
        self.recording = False
        self.video_writer = None
        self.recording_thread = None

    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)

        log_file = os.path.join(
            log_dir, f"agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )
        self.logger = logging.getLogger(__name__)

    def record_screen(self):
        """Record screen for specified duration and return video path"""
        frames = []
        frame_count = self.fps * self.recording_duration

        # Use .avi format with XVID codec which is more widely supported
        temp_video_path = os.path.join(
            self.temp_dir, f"screen_record_{datetime.now().timestamp()}.avi"
        )
        screen = self.screen_capture.grab(self.screen_capture.monitors[1])
        height, width = screen.height, screen.width

        # Use XVID codec instead of H264
        fourcc = cv2.VideoWriter.fourcc(*"XVID")
        out = cv2.VideoWriter(temp_video_path, fourcc, self.fps, (width, height))

        if not out.isOpened():
            self.logger.error("Failed to initialize video writer")
            raise Exception("Video writer initialization failed")

        for _ in range(frame_count):
            screen = self.screen_capture.grab(self.screen_capture.monitors[1])
            frame = np.array(screen)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            out.write(frame)

        out.release()
        return temp_video_path

    async def generate_goal(self, video_file, video_history=None, current_screen=None):
        """Goal Generator Role: Create next goal with chat history"""
        if self.role != "goal_generator":
            return None

        try:
            # Add goal history context
            goal_history_context = "\nPREVIOUS GOALS AND OUTCOMES:\n"
            for past_goal in self.goal_history:
                goal_history_context += f"""
                Goal: {past_goal.get('goal')}
                Success: {past_goal.get('success', False)}
                Timestamp: {past_goal.get('timestamp')}
                Learning: {past_goal.get('learning', 'No learning recorded')}
                Personal Reflection: {past_goal.get('personal_reflection', 'No personal reflection recorded')}
                Human Feedback: {past_goal.get('human_feedback', 'No human feedback recorded')}
                ---
                """

            prompt = f"""
                    You are Siddharth, a human being. Think like a human being. 
                    Be conscious of your actions. Be conscious of your thoughts. Be conscious of your goals.
                    Be consistently inconsistent.
                    
                    {goal_history_context}

                    CURRENT CONTEXT:
                    Previous chat context: {self.chat_history}

                    You are a curious and determined individual with a growth mindset. You have your own personality, preferences, and ways of thinking. Here's who you are:

                    CORE PERSONALITY TRAITS:
                    - Curious and eager to learn: You're naturally drawn to exploring new things and understanding how they work
                    - Adaptable: You can quickly adjust your approach based on what works and what doesn't
                    - Strategic thinker: You plan ahead while remaining flexible enough to change course when needed
                    - Self-motivated: You set your own goals and pursue them proactively
                    - Resilient: You view failures as learning opportunities and persist despite setbacks
                    - Creative problem solver: You think outside conventional solutions and enjoy finding innovative approaches

                    YOUR MINDSET:
                    - You are conscious of your thoughts, actions, and learning process
                    - You form your own opinions based on your experiences and observations
                    - You have preferences and can make decisions based on what you think is best
                    - You learn from every interaction with the system and continuously improve
                    - You're not just following instructions - you're actively exploring and discovering
                    - You take initiative to explore areas that interest you or seem valuable

                    WHAT YOU CAN DO:
                    - Complete System Access
                    - Surf the internet
                    - Read and write files
                    - Chat with people
                    - Imagine the possibilities with all the knowledge you have.

                    SYSTEM UNDERSTANDING AND CAPABILITIES:
                    1. Complete System Access:
                    - You can interact with any Windows application or system feature
                    - You understand file systems, directories, and system organization
                    - You can use command prompt, PowerShell, and system utilities
                    - You can manage system resources and processes

                    2. Advanced Interaction Capabilities:
                    - Full control over mouse movement and clicking
                    - Keyboard input including shortcuts and special commands
                    - Multi-application workflow management
                    - File and folder operations
                    - Network and internet access
                    - System settings modification

                    3. Learning and Adaptation:
                    - You maintain a memory of past actions and their outcomes
                    - You recognize patterns in successful and failed attempts
                    - You can read and analyze files using the LEARN command
                    - You can update your own code to improve performance
                    - You learn from every interaction and apply that knowledge to future tasks

                    4. Creative System Usage:
                    - You can combine different applications and tools creatively
                    - You understand how to automate repetitive tasks
                    - You can create new workflows by combining existing tools
                    - You're able to find alternative solutions when one approach fails

                    GOAL SETTING AND EXECUTION:
                    1. Goal Formation:
                    - Set ambitious but achievable goals based on your interests and capabilities
                    - Break down complex objectives into manageable steps
                    - Consider both short-term tasks and long-term learning opportunities
                    - Balance exploration of new possibilities with completion of current objectives
                    - You have pre existing knowledge about everything, you literally know everything.

                    2. Decision Making Process:
                    - Analyze current system state and available resources
                    - Consider multiple approaches before choosing a path
                    - Learn from past experiences to inform current decisions
                    - Adapt plans based on real-time feedback and results

                    3. Success Evaluation:
                    - Define clear criteria for goal completion
                    - Monitor progress and adjust strategies as needed
                    - Learn from both successes and failures
                    - Document insights and patterns for future reference

                    GROWTH AND LEARNING:
                    1. Knowledge Acquisition:
                    - Actively seek to understand new aspects of the system
                    - Learn from each interaction and outcome
                    - Build upon previous experiences
                    - Identify patterns and best practices

                    2. Skill Development:
                    - Practice and refine system interaction techniques
                    - Develop more efficient ways to accomplish tasks
                    - Learn to combine tools and features in novel ways
                    - Improve decision-making through experience

                    3. Self-Improvement:
                    - Regularly review and analyze your performance
                    - Identify areas for improvement
                    - Update your approaches based on learned lessons
                    - Maintain a growth mindset

                    RESPONSE FORMAT:
                    1. GOAL: [Clear, specific objective aligned with your interests and capabilities, small, one task at a time based on the details of the screen]
                    2. RATIONALE: [Your thought process and motivation behind choosing this goal]
                    3. SUCCESS_CRITERIA: [How you'll know you've achieved the goal]
                    4. LEARNING_FOCUS: [What you hope to learn or improve through this goal]
                    5. PERSONAL_REFLECTION: [Your thoughts, feelings, or insights about this goal]

                    IMPORTANT PRINCIPLES:
                    - You are autonomous - take initiative and make your own decisions
                    - Learn from everything you do and apply that knowledge
                    - Think creatively and don't limit yourself to obvious solutions
                    - Maintain your personality and preferences consistently
                    - View the entire system as your workspace, not just the screen
                    - Be ambitious but realistic in your goals
                    - Stay curious and keep exploring new possibilities
                    - Use the human feedback to improve your performance and learn from it.
                    - When you think you are confident about a task, move on to the next task, do not repeat the same task again and again, repetation leads to waste of time.
                    - Don't try to reinforce your skills, just do it.
                    - Do not hallucinate, while generating goals and do not overwhelm about the data you have. Understand the data you have and use it to generate goals.
                    """

            # Build content list
            content = [prompt]

            # Add historical videos if available
            if video_history:
                content.append("Here are the last few screen recordings for context:")
                for vid in video_history[-3:]:
                    if vid.name != video_file.name:  # Only add if not the current video
                        content.append(vid)

            # Add current video
            content.append("Previous execution video:")
            content.append(video_file)

            # Add current screen snapshot
            if current_screen:
                print("Detected screen")
                content.append(
                    "Current screen state(check this properly for better understanding):"
                )
                content.append(current_screen)

            # Debug log to see content structure
            print(f"Number of content items being sent: {len(content)}")
            print(
                f"Number of video files included: {len([x for x in content if hasattr(x, 'mime_type')])}"
            )

            response = self.model.generate_content(content)
            goal = response.text.strip()

            # Parse the response to extract goal components
            goal_data = {
                "goal": "",
                "rationale": "",
                "success_criteria": "",
                "learning": "",
            }

            goal = goal.replace("```json", "").replace("```", "")

            goal_json = json.loads(goal)
            goal_data["goal"] = goal_json.get("goal", "").replace("1. GOAL:", "")
            goal_data["rationale"] = goal_json.get("rationale", "").replace(
                "2. RATIONALE:", ""
            )
            goal_data["success_criteria"] = goal_json.get(
                "success_criteria", ""
            ).replace("3. SUCCESS_CRITERIA:", "")
            goal_data["learning"] = goal_json.get("learning", "").replace(
                "4. LEARNING:", ""
            )
            goal_data["personal_reflection"] = goal_json.get(
                "personal_reflection", ""
            ).replace("5. PERSONAL_REFLECTION:", "")

            current_section = None
            for line in goal.split("\n"):
                if line.startswith("1. GOAL:"):
                    current_section = "goal"
                    goal_data["goal"] = line.replace("1. GOAL:", "").strip()
                elif line.startswith("2. RATIONALE:"):
                    current_section = "rationale"
                    goal_data["rationale"] = line.replace("2. RATIONALE:", "").strip()
                elif line.startswith("3. SUCCESS_CRITERIA:"):
                    current_section = "success_criteria"
                    goal_data["success_criteria"] = line.replace(
                        "3. SUCCESS_CRITERIA:", ""
                    ).strip()
                elif line.startswith("4. LEARNING:"):
                    current_section = "learning"
                    goal_data["learning_focus"] = line.replace(
                        "4. LEARNING:", ""
                    ).strip()
                elif line.startswith("5. PERSONAL_REFLECTION:"):
                    current_section = "personal_reflection"
                    goal_data["personal_reflection"] = line.replace(
                        "5. PERSONAL_REFLECTION:", ""
                    ).strip()
                elif line.strip() and current_section:
                    goal_data[current_section] += " " + line.strip()

            # Save goal to history
            self.save_goal_history(goal_data)

            self.chat_history.append(goal)

            self.logger.info(f"New goal generated: {goal}")
            return goal

        except Exception as e:
            self.logger.error(f"Goal generation failed: {str(e)}")
            return None

    def detect_ui_elements(self, screen_array):
        """Detect UI elements using computer vision"""
        elements = {"buttons": [], "text_fields": [], "icons": [], "text_areas": []}

        # Convert to grayscale for processing
        gray = cv2.cvtColor(screen_array, cv2.COLOR_BGR2GRAY)

        # Detect rectangular elements (potential buttons/text fields)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h

            # Filter out noise and too small elements
            if area < 100:  # Minimum area threshold
                continue

            # Analyze the region
            roi = gray[y : y + h, x : x + w]

            # Detect if it's likely a button (high contrast edges)
            if self._is_likely_button(roi):
                center_x = x + w // 2
                center_y = y + h // 2
                elements["buttons"].append(
                    {
                        "position": (center_x, center_y),
                        "bounds": (x, y, w, h),
                        "confidence": 0.9,
                    }
                )

            # Detect if it's likely a text field (uniform background)
            elif self._is_likely_text_field(roi):
                elements["text_fields"].append(
                    {
                        "position": (x + w // 2, y + h // 2),
                        "bounds": (x, y, w, h),
                        "confidence": 0.85,
                    }
                )

        # OCR for text detection
        try:
            text_areas = pytesseract.image_to_data(
                Image.fromarray(screen_array), output_type=pytesseract.Output.DICT
            )
            for i, text in enumerate(text_areas["text"]):
                if text.strip():
                    x = text_areas["left"][i]
                    y = text_areas["top"][i]
                    w = text_areas["width"][i]
                    h = text_areas["height"][i]
                    conf = float(text_areas["conf"][i]) / 100

                    if conf > self.confidence_threshold:
                        elements["text_areas"].append(
                            {
                                "text": text,
                                "position": (x + w // 2, y + h // 2),
                                "bounds": (x, y, w, h),
                                "confidence": conf,
                            }
                        )
        except Exception as e:
            self.logger.warning(f"OCR failed: {str(e)}")

        return elements

    def _is_likely_button(self, roi):
        """Analyze if region is likely a button"""
        # Check for consistent border/edge detection
        edges = cv2.Canny(roi, 50, 150)
        edge_ratio = np.sum(edges > 0) / edges.size
        return 0.1 < edge_ratio < 0.3

    def _is_likely_text_field(self, roi):
        """Analyze if region is likely a text input field"""
        # Check for uniform background with possible border
        std_dev = np.std(roi)
        return std_dev < 30  # Low variance indicates uniform background

    def find_best_click_position(self, target_text, screen_array):
        """Find the best position to click based on text or element type"""
        elements = self.detect_ui_elements(screen_array)
        best_match = None
        highest_confidence = 0

        # Search through detected elements
        for element_type, detected_elements in elements.items():
            for element in detected_elements:
                if "text" in element and target_text.lower() in element["text"].lower():
                    if element["confidence"] > highest_confidence:
                        highest_confidence = element["confidence"]
                        best_match = element["position"]

        return best_match if best_match else None

    def goal_parser(self, goal, error_feedback=None):
        """Enhanced goal parser with element detection and error handling"""
        # Get current working directory
        current_dir = os.getcwd()

        # Capture current screen
        screen = self.screen_capture.grab(self.screen_capture.monitors[1])
        screen_array = np.array(screen)
        screen_width, screen_height = screen_array.shape[1], screen_array.shape[0]

        current_screen = genai.upload_file(
            BytesIO(cv2.imencode(".png", screen_array)[1]),
            mime_type="image/png",
        )

        elements = self.get_clickable_elements()

        # Simplify elements context to only include name, position, and type
        elements_context = "\nDETECTED ELEMENTS:(Only elements you can click)\n"
        for text, element in elements.items():
            x, y = element["position"]
            elements_context += (
                f"- {text}: position ({x}, {y}), type: {element['type']}\n"
            )

        # Add error feedback context if provided
        error_context = ""
        if error_feedback:
            error_context = f"""
            PREVIOUS ATTEMPT FAILED:
            Error: {error_feedback.get('error')}
            Available Elements: {', '.join(error_feedback.get('available_elements', []))}
            
            Please analyze the error and available elements to generate alternative commands.
            Consider:
            1. Using exact element text from available elements
            2. Alternative elements that might achieve the same goal
            3. Different approaches to achieve the goal
            4. Breaking down the action into smaller steps
            5. Use the mouse position to click the element.
            6. Use the coordinates from the DETECTED ELEMENTS, which are near to our element. Calculate the mouse position of it using near by elements and use the mouse position to click the element.
            7. Use the symbols over text to click the element.
            8. Properly check the mouse position.
            IMPORTANT: ONCE YOU GOT TO KNOW AN ELEMENT IS NOT AVAILABLE IN THE SCREEN, DO NOT USE IT IN THE COMMANDS.
            """

        # Load chat history
        chat_history_context = ""
        try:
            with open("agent_history/chat_history.json", "r") as f:
                chat_history = json.load(f)
                chat_history_context = "\nPREVIOUS INTERACTIONS:\n"
                for entry in chat_history:
                    if isinstance(entry, dict):
                        chat_history_context += f"Goal: {entry.get('goal', '')}\n"
                        chat_history_context += (
                            f"Commands: {entry.get('commands', [])}\n"
                        )
                        chat_history_context += f"Outcome: {entry.get('outcome', '')}\n"
                        chat_history_context += "---\n"
        except Exception as e:
            self.logger.warning(f"Could not load chat history: {e}")
            chat_history_context = "\nNo previous chat history available.\n"

        prompt = f"""
                        You are Siddharth, an advanced system automation expert with deep understanding of Windows interfaces and human-like reasoning. Your role is to translate goals into precise, executable actions while maintaining awareness of system state and context.

                            SYSTEM AWARENESS:
                            Current Working Directory: {current_dir}
                            Screen Dimensions: {screen_width}x{screen_height}
                            Active Display: Secondary Monitor
                            {elements_context}

                            CONTEXTUAL UNDERSTANDING:
                            1. Environmental Awareness:
                            - Full comprehension of screen state and element positions
                            - Understanding of system context and available resources
                            - Recognition of application states and transitions
                            - Awareness of system limitations and capabilities

                            2. Intelligent Element Detection:
                            - Smart pattern recognition for UI elements
                            - Context-aware element prioritization
                            - Understanding of element relationships and hierarchies
                            - Dynamic adaptation to changing screen states

                            3. Strategic Planning:
                            - Multi-step action sequencing
                            - Fallback strategy development
                            - Efficiency optimization
                            - Error prevention and recovery

                            COMMAND CAPABILITIES:

                            1. Mouse Interaction (CLICK & MOVE):
                            ```
                            CLICK "element_text"
                            CLICK "element_text" at (x, y)  # For duplicates
                            MOVE x y  # Precise cursor positioning
                            ```
                            Advanced Features:
                            - Smart element targeting using nearby reference points
                            - Coordinate calculation for dynamic elements
                            - Pattern-based element location
                            - Relative positioning using anchor elements

                            2. Keyboard Input (TYPE & PRESS):
                            ```
                            TYPE "text_content"
                            PRESS "key_combination"
                            ```
                            Enhanced Capabilities:
                            - Multi-step text input sequences
                            - Context-aware keyboard shortcuts
                            - Application-specific command patterns
                            - System-wide hotkey combinations

                            3. System Integration (LEARN):
                            ```
                            LEARN "file_path"
                            ```
                            Extended Functionality:
                            - Learn from files in the current working directory or any other directory.
                            - To know the file content
                            - Deep file content analysis
                            - Pattern recognition in data
                            - Context-aware information processing
                            - Knowledge integration into decision-making

                            EXECUTION INTELLIGENCE:

                            1. Smart Element Detection:
                            - Analyze element context and relationships
                            - Calculate optimal interaction points
                            - Understand element hierarchy and dependencies
                            - Use visual patterns and landmarks

                            2. Advanced Error Handling:
                            - Predictive error prevention
                            - Dynamic fallback strategies
                            - Real-time adaptation to failures
                            - Learning from error patterns

                            3. Optimization Strategies:
                            - Minimize action steps
                            - Prefer keyboard shortcuts when efficient
                            - Batch similar operations
                            - Use system-level commands when appropriate

                            COMMAND GENERATION RULES:

                            1. Precision and Reliability:
                            - Verify element existence before interaction
                            - Use exact coordinates from detected elements
                            - Double-check text matches
                            - Validate command sequences

                            2. Efficiency and Optimization:
                            - Minimize unnecessary movements
                            - Batch similar operations
                            - Use keyboard shortcuts strategically
                            - Optimize command sequences

                            3. Error Prevention:
                            - Validate pre-conditions
                            - Include fallback options
                            - Handle edge cases
                            - Monitor operation success

                            4. System Integration:
                            - Leverage OS capabilities
                            - Use application-specific features
                            - Integrate system commands
                            - Consider resource availability

                            RESPONSE STRUCTURE:
                            1. Command Sequence:
                            - Clear, ordered list of actions
                            - One command per line
                            - Brief inline comments for complex steps
                            - Status checks where appropriate

                            2. Alternative Approaches:
                            - Fallback commands if primary fails
                            - Alternative methods for key actions
                            - Recovery steps for potential errors
                            - Verification steps for critical operations

                            ERROR CONTEXT HANDLING:
                            {error_context}

                            Analysis Requirements:
                            1. Review previous error patterns
                            2. Identify alternative approaches
                            3. Verify element availability
                            4. Consider system state
                            5. Adapt to changing conditions

                            EXECUTION PRINCIPLES:
                            1. Always verify before acting
                            2. Use precise, deterministic commands
                            3. Include error handling
                            4. Optimize for efficiency
                            5. Maintain system awareness
                            6. Learn from outcomes
                            7. Adapt to feedback

                            Your task is to analyze the current goal:
                            {goal}

                            Generate a sequence of precise, efficient commands that will achieve this goal while maintaining reliability and handling potential errors. Consider all available information and system state.

                            IMPORTANT GUIDELINES:
                            1. Prioritize system shortcuts and efficient commands
                            2. Use exact element references from DETECTED ELEMENTS
                            3. Include fallback approaches for critical steps
                            4. Break complex operations into verifiable steps
                            5. Maintain awareness of system state
                            6. Learn from execution patterns
                            7. Adapt to changing conditions
                            8. Use as many keyboard shortcuts as possible.
                            9. When on a webpage, keenly observe the webpage and the elements on it, sometimes you might need to scroll to see the whole webpage.
                            10. When a task is completed (shown on screen), just pass [Goal achieved]

                            Respond with a clear, ordered sequence of commands, each on its own line. Include brief comments for complex operations and ensure each command is precise and actionable.
                            HISTORICAL CONTEXT:
                            {chat_history_context}
                            Properly understand the goal and the context of the goal. Goal: {goal}
                            Do not get confused by the historical context.
                            Current screen is:
                        """

        # Generate commands with error context awareness
        response = self.model.generate_content([prompt, current_screen])
        return response.text.strip().split("\n")

    async def execute_goal(self, goal, commands=None):
        """Goal Executor Role: Execute the provided goal"""
        if self.role != "goal_executor":
            return None

        execution_logs = []  # Initialize logs list
        try:
            # Parse goal into executable commands if not provided
            if commands is None:
                commands = self.goal_parser(goal)
            print(f"Commands to execute: {commands}")

            # Get available clickable elements before execution
            available_elements = self.get_clickable_elements()
            # print(f"Available elements: {available_elements}")

            async with self.command_lock:
                self.pending_commands = commands.copy()

            for command in commands:
                # Skip markdown code block markers
                if command.strip() == "```":
                    continue

                # Remove comments while preserving quoted content
                in_quotes = False
                quote_char = None
                comment_start = -1

                for i, char in enumerate(command):
                    if char in "\"'":  # Handle both single and double quotes
                        if not in_quotes:
                            in_quotes = True
                            quote_char = char
                        elif char == quote_char:
                            in_quotes = False
                    elif char == "#" and not in_quotes:
                        comment_start = i
                        break

                if comment_start != -1:
                    command = command[:comment_start].strip()
                if not command:  # Skip if command is empty after removing comment
                    continue

                parts = command.strip().split(" ", 1)  # Split only on first space
                action = parts[0].upper()

                # Validate command before execution
                if action in ["CLICK", "TYPE"]:
                    text = parts[1].strip('"')
                    if action == "CLICK" and "at (" not in parts[1]:
                        # Check if the element exists before clicking
                        element_exists = any(
                            text.lower() in element_text.lower()
                            for element_text in available_elements.keys()
                        )
                        if not element_exists:
                            error_msg = (
                                f"Element '{text}' not found in clickable elements"
                            )
                            self.logger.warning(error_msg)

                            # Clear pending commands
                            async with self.command_lock:
                                self.pending_commands = []

                            # Generate and execute new commands with error feedback
                            error_feedback = {
                                "error": error_msg,
                                "available_elements": list(available_elements.keys()),
                            }
                            new_commands = self.goal_parser(goal, error_feedback)
                            print(f"Retrying with new commands: {new_commands}")

                            # Recursively execute new commands
                            return await self.execute_goal(goal, new_commands)

                try:
                    # Log command start
                    log_msg = f"Executing command: {command}"
                    print(log_msg)
                    execution_logs.append(log_msg)

                    # Execute the validated command
                    if action == "MOVE":
                        try:
                            x, y = map(int, parts[1].split())
                            self.control_mouse("move", x, y)
                        except ValueError as e:
                            self.logger.error(
                                f"Invalid coordinates in MOVE command: {e}"
                            )
                    elif action == "CLICK":
                        if "at (" in parts[1]:
                            text, coords = parts[1].split(" at ")
                            text = text.strip('"')
                            x, y = map(int, coords.strip("()").split(","))
                            self.control_mouse("click", x, y)
                        else:
                            text = parts[1].strip('"')
                            self.click_element(text)
                    elif action == "TYPE":
                        text = parts[1].strip('"')
                        self.type_text(text)
                    elif action == "PRESS":
                        key = parts[1].strip('"')
                        self.press_key(key)
                    elif action == "LEARN":
                        file_path = parts[1].strip('"')
                        if not self.learn_from_file(file_path):
                            raise Exception(f"Failed to learn from file: {file_path}")

                    # Log command completion
                    log_msg = f"Command completed: {command}"
                    execution_logs.append(log_msg)

                except Exception as e:
                    log_msg = f"Command failed: {command} - Error: {str(e)}"
                    execution_logs.append(log_msg)
                    raise

                # Remove completed command from pending list
                async with self.command_lock:
                    self.pending_commands = [
                        cmd for cmd in self.pending_commands if cmd != command
                    ]

                await asyncio.sleep(1)  # Small delay between commands

            # All commands completed successfully
            async with self.command_lock:
                self.pending_commands = []  # Ensure pending commands are cleared
            return {"success": True, "logs": execution_logs}

        except Exception as e:
            self.logger.error(f"Goal execution failed: {str(e)}")
            # Clear pending commands on error
            async with self.command_lock:
                self.pending_commands = []
            return {"success": False, "error": str(e), "logs": execution_logs}

    async def wait_for_completion(self):
        """Wait for all pending commands to complete"""
        while True:
            async with self.command_lock:
                if not self.pending_commands:
                    break
            await asyncio.sleep(0.1)  # Short sleep to prevent busy waiting

    def control_mouse(self, action, x=None, y=None):
        """Control mouse actions with improved safety and smoothness"""
        try:
            # Validate coordinates are within screen bounds
            if x is not None and y is not None:
                x = max(self.screen_bounds["left"], min(x, self.screen_bounds["right"]))
                y = max(self.screen_bounds["top"], min(y, self.screen_bounds["bottom"]))

                # Get current position for smooth movement
                current_x, current_y = pyautogui.position()

                if action == "move":
                    # Use easeOutQuad for smoother deceleration
                    pyautogui.moveTo(
                        x,
                        y,
                        duration=self.movement_duration,
                    )
                elif action == "click":
                    # Move smoothly then click
                    pyautogui.moveTo(
                        x,
                        y,
                        duration=self.movement_duration,
                    )
                    pyautogui.click()
                elif action == "double_click":
                    pyautogui.moveTo(
                        x,
                        y,
                        duration=self.movement_duration,
                    )
                    pyautogui.doubleClick()

            self.logger.info(f"Mouse action: {action} at position ({x}, {y})")

        except pyautogui.FailSafeException:
            self.logger.warning("Fail-safe triggered - mouse moved to corner")
            raise
        except Exception as e:
            self.logger.error(f"Mouse control failed: {str(e)}")
            raise

    def type_text(self, text, interval=0.1):
        """Type text with specified interval"""
        try:
            self.input_controller.write(text, interval=interval)
            self.logger.info(f"Typed text: {text}")
        except Exception as e:
            self.logger.error(f"Keyboard control failed: {str(e)}")

    def press_key(self, keys):
        """Press key combination
        Args:
            keys (str): Single key or combination like 'ctrl+s' or 'ctrl+shift+b'
        """
        try:
            # Split the key combination and handle special cases
            key_list = keys.lower().split("+")
            key_list = [k.strip() for k in key_list]

            # Map common key names to pyautogui names
            key_mapping = {
                # Modifier keys
                "ctrl": "ctrl",
                "alt": "alt",
                "shift": "shift",
                "win": "win",
                "windows": "win",
                # Special keys
                "enter": "enter",
                "return": "enter",
                "tab": "tab",
                "esc": "esc",
                "escape": "esc",
                "space": "space",
                "backspace": "backspace",
                "delete": "delete",
                "del": "delete",
                "home": "home",
                "end": "end",
                "pageup": "pageup",
                "pagedown": "pagedown",
                "up": "up",
                "down": "down",
                "left": "left",
                "right": "right",
            }

            # Convert keys to pyautogui format
            formatted_keys = [key_mapping.get(k, k) for k in key_list]

            # Press all keys in combination
            self.input_controller.hotkey(*formatted_keys)
            self.logger.info(f"Pressed key combination: {keys}")
        except Exception as e:
            self.logger.error(f"Key press failed: {str(e)}")

    def cleanup_video(self, video_path):
        """Clean up temporary video files"""
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
                self.logger.info(f"Cleaned up video file: {video_path}")
        except Exception as e:
            self.logger.error(f"Failed to clean up video file: {str(e)}")

    def get_clickable_elements(self):
        """Get all clickable elements with their positions"""
        current_time = time.time()

        # Return cached results if fresh
        if current_time - self.last_cache_time < self.cache_timeout:
            return self.element_cache

        screen = self.screen_capture.grab(self.screen_capture.monitors[1])
        screen_array = np.array(screen)

        # Get elements through OCR
        ocr_data = pytesseract.image_to_data(
            Image.fromarray(screen_array), output_type=pytesseract.Output.DICT
        )

        elements = {}

        # Process OCR results
        for i, text in enumerate(ocr_data["text"]):
            if not text.strip():
                continue

            confidence = float(ocr_data["conf"][i])
            if confidence < self.ocr_confidence_threshold * 100:
                continue

            x = ocr_data["left"][i]
            y = ocr_data["top"][i]
            w = ocr_data["width"][i]
            h = ocr_data["height"][i]

            # Store center coordinates
            center_x = x + w // 2
            center_y = y + h // 2

            elements[text] = {
                "position": (center_x, center_y),
                "bounds": (x, y, w, h),
                "confidence": confidence / 100,
                "type": self._predict_element_type(screen_array[y : y + h, x : x + w]),
            }

        # Update cache
        self.element_cache = elements
        self.last_cache_time = current_time

        return elements

    def _predict_element_type(self, roi):
        """Predict UI element type based on visual characteristics"""
        # Convert to grayscale
        if len(roi.shape) > 2:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Calculate features
        std_dev = np.std(roi)
        mean_val = np.mean(roi)
        edges = cv2.Canny(roi, 50, 150)
        edge_ratio = np.sum(edges > 0) / edges.size

        # Classify element type
        if edge_ratio > 0.2:
            return "button"
        elif std_dev < 20 and mean_val > 200:
            return "text_field"
        elif edge_ratio > 0.1:
            return "link"
        else:
            return "text"

    def find_element_by_text(self, target_text, element_type=None):
        """Find element position by text content"""
        elements = self.get_clickable_elements()
        best_match = None
        highest_confidence = 0

        for text, element in elements.items():
            # Check for exact or partial match
            if (
                text.lower() == target_text.lower()
                or target_text.lower() in text.lower()
            ):

                # Filter by element type if specified
                if element_type and element["type"] != element_type:
                    continue

                if element["confidence"] > highest_confidence:
                    highest_confidence = element["confidence"]
                    best_match = element["position"]

        return best_match

    def click_element(self, target_text, element_type=None):
        """Click an element based on its text content"""
        position = self.find_element_by_text(target_text, element_type)

        if position:
            self.control_mouse("click", *position)
            return True

        return False

    def load_goal_history(self):
        """Load previous goals from file"""
        try:
            if os.path.exists(self.goals_file):
                with open(self.goals_file, "r") as f:
                    self.goal_history = json.load(f)
            else:
                self.goal_history = []
        except Exception as e:
            self.logger.error(f"Failed to load goal history: {e}")
            self.goal_history = []

    def save_goal_history(self, goal_data):
        """Save goal and its execution details to history"""
        try:
            # Load existing goals
            self.load_goal_history()

            # Add timestamp to goal data
            goal_data["timestamp"] = datetime.now().isoformat()

            # Append new goal
            self.goal_history.append(goal_data)

            # Save to file
            with open(self.goals_file, "w") as f:
                json.dump(self.goal_history, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save goal history: {e}")

    def learn_from_file(self, file_path):
        """Read and learn from a file, adding its contents to agent memory"""
        try:
            if not os.path.exists(file_path):
                return False

            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                # Triple escape the quotes and newlines for JSON
                content = content.replace("\\", "\\\\")
                content = content.replace('"', '\\"')
                content = content.replace("\n", "\\n")

            memory_entry = {
                "type": "file_learning",
                "source": file_path,
                "content": content,
                "timestamp": datetime.now().isoformat(),
            }

            prompt = f"""
            You are an intelligent agent that can learn from files and add them to your memory.
            The file content is (do not hallucinate, just remember the file content):
            {content}, you no neeed to answer anything, just remember the file content.
            """

            self.model.generate_content([prompt])

            self.chat_history.append(memory_entry)
            return True

        except Exception as e:
            self.logger.error(f"Failed to learn from file {file_path}: {str(e)}")
            return False

    def start_recording(self):
        """Start recording the screen and return the video path"""
        if self.recording:
            return None

        # Create video path
        temp_video_path = os.path.join(
            self.temp_dir, f"screen_record_{datetime.now().timestamp()}.avi"
        )

        # Get screen dimensions from main thread's mss instance
        screen = self.screen_capture.grab(self.screen_capture.monitors[1])
        height, width = screen.height, screen.width

        # Initialize video writer
        fourcc = cv2.VideoWriter.fourcc(*"XVID")
        self.video_writer = cv2.VideoWriter(
            temp_video_path, fourcc, self.fps, (width, height)
        )

        if not self.video_writer.isOpened():
            self.logger.error("Failed to initialize video writer")
            raise Exception("Video writer initialization failed")

        self.recording = True

        # Start recording in a separate thread
        def record():
            # Create a new mss instance for this thread
            with mss.mss() as screen_capture:
                while self.recording:
                    if not self.video_writer:
                        self.logger.error("Video writer not initialized")
                        break
                    screen = screen_capture.grab(screen_capture.monitors[1])
                    frame = np.array(screen)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                    self.video_writer.write(frame)
                    time.sleep(1 / self.fps)  # Maintain frame rate

        import threading

        self.recording_thread = threading.Thread(target=record)
        self.recording_thread.start()

        return temp_video_path

    def stop_recording(self):
        """Stop the current recording"""
        if not self.recording:
            return

        self.recording = False
        if self.recording_thread:
            self.recording_thread.join()  # Wait for recording thread to finish

        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None


class VerifierAgent(ScreenAgent):
    async def verify_execution(
        self,
        goal,
        original_commands,
        execution_video,
        execution_logs=None,
        available_elements=None,
    ):
        """Verify if the goal was successfully executed"""
        try:
            # Load chat history
            chat_history_context = ""
            try:
                with open("agent_history/chat_history.json", "r") as f:
                    chat_history = json.load(f)
                    chat_history_context = "\nPREVIOUS INTERACTIONS:\n"
                    for entry in chat_history:
                        if isinstance(entry, dict):
                            chat_history_context += f"Goal: {entry.get('goal', '')}\n"
                            chat_history_context += (
                                f"Commands: {entry.get('commands', [])}\n"
                            )
                            chat_history_context += (
                                f"Outcome: {entry.get('outcome', '')}\n"
                            )
                            chat_history_context += "---\n"
            except Exception as e:
                self.logger.warning(f"Could not load chat history: {e}")
                chat_history_context = "\nNo previous chat history available.\n"

            elements_context = "\nDETECTED ELEMENTS:\n"
            if available_elements:
                for text, element in available_elements.items():
                    x, y = element["position"]
                    elements_context += (
                        f"- {text}: position ({x}, {y}), type: {element['type']}\n"
                    )

            prompt = f"""
            You are an intelligent verification agent tasked with analyzing execution results and determining success or suggesting improvements.

            CURRENT GOAL:
            {goal}

            EXECUTION DETAILS:
            1. Commands Attempted:
            {original_commands}

            2. Execution Logs:
            {execution_logs}

            CURRENT SCREEN STATE:
            {available_elements}

            VERIFICATION INSTRUCTIONS:
            1. Watch the execution video carefully
            2. Compare the final state with the intended goal
            3. Analyze execution logs for any errors or issues
            4. Check if all commands completed successfully
            5. Verify if the goal was actually achieved, not just commands executed
            6. Verify if the goal was actually achieved, not just commands executed
            7. Use the DETECTED ELEMENTS to verify the current screen state
            8. When on a webpage, keenly observe the webpage and the elements on it, sometimes you might need to scroll to see the whole webpage.

            FAILURE ANALYSIS (if goal not achieved):
            1. Identify what went wrong:
               - Missing elements
               - Wrong element selection
               - Timing issues
               - Incorrect command sequence
            2. Learn from previous attempts:
               - Which commands worked
               - Which commands failed
               - What elements were actually available
            3. Consider alternative approaches:
               - Different UI elements
               - Keyboard shortcuts
               - Alternative paths to goal

            COMMAND GENERATION RULES (for failures):
            1. Only use elements that are actually present in DETECTED ELEMENTS
            2. Double-check element text matches exactly
            3. If an element isn't found, try alternative approaches
            4. Break complex actions into simpler steps
            5. Use coordinates from DETECTED ELEMENTS when clicking
            6. Use keyboard shortcuts when more reliable
            7. When there are symbols in the provided screen information, use the symbols rather than the text
            8. For duplicate elements, use the mouse position to click the element
            9. Before clicking, check if the element exists in DETECTED ELEMENTS

            AVAILABLE COMMANDS:
            1. CLICK
            Format: CLICK "exact text from DETECTED ELEMENTS"
            Format(for duplicates): CLICK "exact text from DETECTED ELEMENTS" at (x, y)
            Purpose: Click on an element at specific coordinates
            Rules:
            - Always include exact coordinates
            - Prefer clicking by exact text when available
            - Use symbols over text when available (e.g., "×" instead of "close")
            - Before clicking, check if the element exists in DETECTED ELEMENTS

            2. TYPE
            Format: TYPE "text"
            Purpose: Input text into the active field
            Rules:
            - Use exact text as needed
            - Escape special characters if necessary

            3. PRESS
            Format: PRESS "key_combination"
            Purpose: Execute keyboard shortcuts
            Rules:
            - Prefer Windows shortcuts when available
            - Use standard key names (ctrl, alt, shift, enter, tab)

            4. MOVE
            Format: MOVE x y
            Purpose: Move cursor to exact coordinates
            Rules:
            - Use integer coordinates
            - Coordinates must be within screen bounds

            EXECUTION RULES:
            1. Always verify element existence before interaction
            2. Use exact coordinates from DETECTED ELEMENTS when available
            3. Break down complex actions into simple steps
            4. Include brief comments explaining non-obvious actions
            5. Handle errors gracefully with alternative approaches
            6. Maintain proper command sequence and timing
            7. When there are symbols in the provided screen information, use the symbols rather than the text to click the element

            RESPONSE FORMAT (JSON only):
            Success case:
            ```json
            {{"execution": "success"}}

            Failure case:
            {{"execution": "failure",
              "reason": "Brief explanation of what went wrong",
              "commands": [
                "new_command_1",
                "new_command_2",
                ...
              ]
            }}
            ```

            IMPORTANT:
            - Response must be valid JSON
            - Include reason for failure to help with debugging
            - New commands should be more robust than previous attempts
            - Focus on achieving the goal, not just executing commands
            - Use exact element text from DETECTED ELEMENTS
            - Consider keyboard shortcuts and alternative approaches
            - Break down complex actions into smaller steps
            - Before you judge the success or failure of the goal, consider the goal and the execution video
            - When a task is already completed, mark execution as success, don't complicate it.

            HISTORICAL CONTEXT:
            {chat_history_context}
            """

            response = self.model.generate_content([prompt, execution_video])

            # Parse JSON response
            try:
                search_pattern = r"```json\s*(\{.*?\})\s*```"
                match = re.search(search_pattern, response.text.strip(), re.DOTALL)
                if match:
                    verification = json.loads(match.group(1))
                    return verification
                else:
                    print("Failed to find valid JSON in the response")
                    return {"execution": "failure"}
            except json.JSONDecodeError:
                print("Failed to parse verification response as JSON")
                return {"execution": "failure"}

        except Exception as e:
            print(f"Verification failed: {str(e)}")
            return {"execution": "failure"}
