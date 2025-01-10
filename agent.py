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

        self.final_goal = "To gain one lakh followers on X (Twitter)."

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

        self.dairy = self.load_dairy()
        self.file_memory = self.load_file_memory()
        self.key_memory = self.load_key_memory()

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

    def load_dairy(self):
        """Load dairy from file at startup"""
        try:
            with open("agent_history/dairy.txt", "r") as f:
                dairy = f.read()
        except FileNotFoundError:
            # Create file with initial content if it doesn't exist
            dairy = (
                "Initial dairy entry. Starting to record my thoughts and experiences."
            )
            with open("agent_history/dairy.txt", "w") as f:
                f.write(dairy)
        return dairy

    def load_file_memory(self):
        """Load file memory from file at startup"""
        try:
            with open("agent_history/file_memory.txt", "r") as f:
                file_memory = f.read()
        except FileNotFoundError:
            # Create file with initial content if it doesn't exist
            file_memory = "Initial file memory entry. Starting to record my thoughts and experiences."
            with open("agent_history/file_memory.txt", "w") as f:
                f.write(file_memory)
        return file_memory

    def load_key_memory(self):
        """Load key memory from file at startup"""
        with open("agent_history/key_memory.txt", "r") as f:
            key_memory = f.read()
        return key_memory

    def save_file_memory(self, content):
        """Save file memory to file"""
        with open("agent_history/file_memory.txt", "w") as f:
            f.write(content)

    async def generate_goal(
        self, video_file=None, video_history=None, current_screen=None
    ):
        """Goal Generator Role: Create next goal with enhanced context"""
        if self.role != "goal_generator":
            return None

        try:
            # Load learning context
            try:
                with open("agent_history/learning.json", "r") as f:
                    learning_data = json.load(f)
            except FileNotFoundError:
                learning_data = {"experiences": [], "patterns": [], "skills": []}

            try:
                with open("agent_history/personality.json", "r") as f:
                    personality_data = json.load(f)
            except FileNotFoundError:
                personality_data = {
                    "traits": {},
                    "interests": [],
                    "learned_behaviors": [],
                    "goals": [],
                }

            # Add goal history context
            goal_history_context = "\nPREVIOUS GOALS AND OUTCOMES:\n"
            for past_goal in self.goal_history:
                goal_history_context += f"""
                Goal: {past_goal.get('goal')}                
                Human Feedback: {past_goal.get('human_feedback', 'No human feedback recorded')}
                Your Final Goal: {past_goal.get('your_final_goal', 'Gain one lakh followers on X (Twitter)')}
                ---
                """

            past_five_goals = self.goal_history[-5:]

            prompt = f"""
            You are Siddharth, a human being with evolving personality and learning capabilities.

            LEARNING CONTEXT:
            Recent Experiences: {json.dumps(learning_data['experiences'][-5:], indent=2)}
            Identified Patterns: {json.dumps(learning_data['patterns'][-5:], indent=2)}
            Acquired Skills: {json.dumps(learning_data['skills'], indent=2)}

            PERSONALITY PROFILE:
            Traits: {json.dumps(personality_data['traits'], indent=2)}
            Interests: {json.dumps(personality_data['interests'], indent=2)}
            Learned Behaviors: {json.dumps(personality_data['learned_behaviors'], indent=2)}
            Current Goals: {json.dumps(personality_data['goals'], indent=2)}

            {goal_history_context}

            CURRENT CONTEXT:
            Previous chat context: {self.chat_history}

            File memory(read this file to understand the file system and the files you have access to):
            {self.file_memory}

            You are a curious and determined individual with a growth mindset. Your personality and capabilities evolve based on experiences:

            CORE PERSONALITY TRAITS:
            - Curiosity Level: {personality_data['traits'].get('curiosity', 0.8)}
            - Persistence Level: {personality_data['traits'].get('persistence', 0.7)}
            - Adaptability Level: {personality_data['traits'].get('adaptability', 0.6)}
            - Creativity Level: {personality_data['traits'].get('creativity', 0.7)}

            YOUR MINDSET:
            - You learn from every interaction and experience
            - Your personality evolves based on outcomes
            - You form preferences based on past experiences
            - You maintain awareness of your growth journey
            - You adapt your approach based on learned patterns
            - You leverage your acquired skills effectively

            WHAT YOU CAN DO:
            - Complete System Access
            - Surf the internet
            - Read and write files
            - Chat with people
            - Learn from experiences
            - Update your personality
            - Maintain a diary of experiences

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

            Your dairy:
            {self.dairy}

            Your key memory:
            {self.key_memory}

            IMPORTANT PRINCIPLES:
            - Learn from every experience and update your approach
            - Consider your personality traits when setting goals
            - Use your acquired skills effectively
            - Build upon identified patterns
            - Maintain consistency with your evolving personality
            - Balance exploration with known capabilities
            - Consider your current interests and goals
            - Use past experiences to inform decisions
            - Write reflective diary entries about your experiences

            PREVIOUS GOALS AND OUTCOMES:
            {past_five_goals}

            Your final goal is: {self.final_goal}
            Every goal you generate should be aligned with this final goal.

            RESPONSE FORMAT (Do not add any invalid "\"escape characters):
            ```json
            {{
                "GOAL": [Clear, specific objective aligned with your current capabilities]
                "RATIONALE": [How this goal builds on your experiences and learning and reach the final goal]
                "SUCCESS_CRITERIA": [How you'll know you've achieved the goal]
                "PERSONAL_REFLECTION": [How this goal reflects your growth journey]
                "GOAL_SIMILARITY": [Analysis of similarity to previous goals]
            }}
            ```

            Instructions for response:
            - Consider your learning history when setting goals
            - Align goals with your personality traits
            - Build upon identified patterns
            - Leverage acquired skills
            - Maintain personality consistency
            - Write meaningful diary entries
            - Learn from every interaction
            - Never ever repeat the same goal, always generate a new goal.
            """

            # Build content list
            content = [prompt]

            # Add historical videos if available
            if video_history and video_file:
                content.append("Here are the last few screen recordings for context:")
                for vid in video_history[-3:]:
                    if vid.name != video_file.name:
                        content.append(vid)

            if video_file:
                content.append("Previous execution video:")
                content.append(video_file)

            if current_screen:
                content.append("Current screen state:")
                content.append(current_screen)

            response = self.model.generate_content(content)
            goal = response.text.strip()

            # Parse the response to extract goal components
            goal_data = {
                "goal": "",
                "rationale": "",
                "success_criteria": "",
                "personal_reflection": "",
                "goal_similarity": "",
            }

            goal = re.search(r"```json(.*?)```", goal, re.DOTALL)
            if not goal:
                return None

            try:
                goal_json = json.loads(goal.group(1))
                print(f"goals: {goal_json}")

                # Map the correct keys from the response
                goal_data["goal"] = goal_json.get("GOAL", "")
                goal_data["rationale"] = goal_json.get("RATIONALE", "")
                goal_data["success_criteria"] = goal_json.get("SUCCESS_CRITERIA", "")
                goal_data["learning"] = goal_json.get("LEARNING_FOCUS", "")
                goal_data["personal_reflection"] = goal_json.get(
                    "PERSONAL_REFLECTION", ""
                )

                self.save_goal_history(goal_data)

                return goal_data

            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {e}")
                return None

        except Exception as e:
            self.logger.error(f"Goal generation failed: {str(e)}")
            return None

        # except Exception as e:
        #     self.logger.error(f"Goal generation failed: {str(e)}")
        #     return None

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
                            Available Elements: {elements_context}

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

                            1. Mouse Interaction (CLICK, RIGHT_CLICK & MOVE):
                            ```
                            CLICK "element_text"
                            CLICK "element_text" at (x, y)  # For duplicates
                            RIGHT_CLICK "element_text"  # For context menus
                            RIGHT_CLICK "element_text" at (x, y)  # For duplicate elements
                            MOVE x y  # Precise cursor positioning
                            ```
                            Advanced Features:
                            - Smart element targeting using nearby reference points
                            - Coordinate calculation for dynamic elements
                            - Pattern-based element location
                            - Relative positioning using anchor elements
                            - Context menu interactions with right-click

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

                            KEY MEMORY:
                            {self.key_memory}

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
                            Properly understand the goal and the context of the goal. 
                            Break down the goal into smaller tasks and execute them one by one. Use the available elements and the screen details to achieve the goal.

                            Sometimes the goal might be acheived partially, then you need to continue the goal from where it left off. After writing the commands in the format, you need to reason about the commands you wrote and how can they acheive the goal.
                            Goal: {goal}
                            Do not get confused by the historical context. AND NEVER RELAY ON WHAT YOU SEEN IN THE SCREEN, CROSS VERIFY THE ELEMENTS AND THE COMMANDS. CHECK AVAILABLE ELEMENTS AND THE COMMANDS.
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
                    if action == "CLICK":
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
                        try:
                            if not self.learn_from_file(file_path):
                                raise Exception(
                                    f"Failed to learn from file: {file_path}"
                                )
                        except FileNotFoundError as e:
                            # Generate specific error feedback for file not found
                            error_feedback = {
                                "error": str(e),
                                "error_type": "file_not_found",
                                "attempted_path": file_path,
                                "available_elements": list(
                                    self.get_clickable_elements().keys()
                                ),
                                "current_directory": os.getcwd(),
                                "suggested_paths": [
                                    os.path.join(os.getcwd(), file_path),
                                    os.path.abspath(file_path),
                                    # Add other potential paths
                                ],
                            }
                            # Clear pending commands
                            async with self.command_lock:
                                self.pending_commands = []

                            # Get new commands with error feedback
                            new_commands = self.goal_parser(goal, error_feedback)
                            print(
                                f"Retrying with new commands after file not found: {new_commands}"
                            )
                            return await self.execute_goal(goal, new_commands)

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
            return {
                "success": False,
                "error": str(e),
                "logs": execution_logs,
                "error_type": (
                    "file_not_found"
                    if isinstance(e, FileNotFoundError)
                    else "execution_error"
                ),
            }

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
                    pyautogui.moveTo(x, y, duration=self.movement_duration)
                elif action == "click":
                    pyautogui.moveTo(x, y, duration=self.movement_duration)
                    pyautogui.click()
                elif action == "right_click":  # Add right-click support
                    pyautogui.moveTo(x, y, duration=self.movement_duration)
                    pyautogui.rightClick()
                elif action == "double_click":
                    pyautogui.moveTo(x, y, duration=self.movement_duration)
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

    def click_element(self, target_text, element_type=None, right_click=False):
        """Click an element based on its text content"""
        position = self.find_element_by_text(target_text, element_type)

        if position:
            self.control_mouse("right_click" if right_click else "click", *position)
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
                raise FileNotFoundError(f"File not found: {file_path}")

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

            self.save_file_memory(str(memory_entry))

            self.model.generate_content([prompt])

            self.chat_history.append(memory_entry)
            return True

        except FileNotFoundError as e:
            error_msg = str(e)
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        except Exception as e:
            error_msg = f"Failed to learn from file {file_path}: {str(e)}"
            self.logger.error(error_msg)
            raise Exception(error_msg)

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
            Format: TYPE "text" + PRESS "enter" (if needed)
            Purpose: Input text into the active field
            Rules:
            - Include PRESS "enter" in the same command if needed
            - Use exact text as needed
            - Escape special characters if necessary

            3. PRESS
            Format: PRESS "key_combination"
            Purpose: Execute keyboard shortcuts
            Rules:
            - Use keyboard shortcuts from KEY MEMORY
            - Use standard key names (ctrl, alt, shift, enter, tab)
            - Combine multiple keys with +

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
            - Be strict with your judgement, do not be lenient with your judgement

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
                    print(f"verification: {verification}")
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


class GoalSimplifierAgent(ScreenAgent):
    async def simplify_goal(
        self, goal, current_screen=None, available_elements=None, previous_goals=None
    ):
        """Simplify complex goals into focused, actionable sub-goals with dynamic updates"""
        try:
            elements_context = "\nDETECTED ELEMENTS:\n"
            if available_elements:
                for text, element in available_elements.items():
                    x, y = element["position"]
                    elements_context += (
                        f"- {text}: position ({x}, {y}), type: {element['type']}\n"
                    )

            previous_goals_context = "\nPREVIOUS SUB-GOALS AND THEIR STATUS:\n"
            completed_goals = []
            failed_goals = []
            if previous_goals:
                for idx, prev_goal in enumerate(previous_goals):
                    status = prev_goal.get("status", "pending")
                    if status == "completed":
                        completed_goals.append(prev_goal["goal"])
                    elif status == "failed":
                        failed_goals.append(prev_goal["goal"])
                    previous_goals_context += (
                        f"{idx + 1}. {prev_goal['goal']} - Status: {status}\n"
                    )

            prompt = f"""
            You are an intelligent goal simplification agent. Your task is to analyze the main goal and generate the next set of sub-goals based on current progress.

            MAIN GOAL:
            {goal}

            CURRENT PROGRESS:
            Completed Sub-goals: {json.dumps(completed_goals, indent=2)}
            Failed Sub-goals: {json.dumps(failed_goals, indent=2)}
            {previous_goals_context}

            CURRENT SCREEN STATE:
            {elements_context}

            KEY MEMORY:
            {self.key_memory}

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
            Format: TYPE "text" + PRESS "enter" (if needed)
            Purpose: Input text into the active field
            Rules:
            - Include PRESS "enter" in the same command if needed
            - Use exact text as needed
            - Escape special characters if necessary

            3. PRESS
            Format: PRESS "key_combination"
            Purpose: Execute keyboard shortcuts
            Rules:
            - Use keyboard shortcuts from KEY MEMORY
            - Use standard key names (ctrl, alt, shift, enter, tab)
            - Combine multiple keys with +

            4. MOVE
            Format: MOVE x y
            Purpose: Move cursor to exact coordinates
            Rules:
            - Use integer coordinates
            - Coordinates must be within screen bounds

            DYNAMIC PLANNING RULES:
            1. Use KEY MEMORY for optimal command sequences
            2. Break complex goals into logical sub-goals
            3. For failed sub-goals:
               - Check KEY MEMORY for alternative approaches
               - Break down into smaller steps
               - Use different interaction patterns
            4. Never repeat completed sub-goals
            5. Build on successful steps
            6. Consider current screen state
            7. Use available elements
            8. Follow interaction patterns from KEY MEMORY
            9. Flag completion when goal is achieved

            VERIFICATION RULES:
            1. Each sub-goal must be verifiable
            2. Use success criteria from KEY MEMORY
            3. Consider screen elements
            4. Provide clear indicators
            5. Include fallback verification

            RESPONSE FORMAT:
            ```json
            {{
                "sub_goals": [
                    {{
                        "goal": "Clear, focused action statement",
                        "success_criteria": "Specific, verifiable condition",
                        "fallback_commands": [
                            "Alternative commands from KEY MEMORY",
                            "Different interaction patterns"
                        ],
                        "prerequisites": ["Required conditions"],
                        "depends_on": ["IDs of dependent sub-goals"]
                    }}
                ],
                "main_goal_completed": false,
                "rationale": "Explanation using KEY MEMORY patterns",
                "progress_percentage": 0,
                "next_milestone": "Next major milestone",
                "adaptation_notes": "How this adapts to previous results"
            }}
            ```

            IMPORTANT RULES:
            1. Always check KEY MEMORY for optimal approaches
            2. Keep sub-goals focused but not oversimplified
            3. Include PRESS "enter" within TYPE commands when needed
            4. Use exact element text from DETECTED ELEMENTS
            5. Follow interaction patterns from KEY MEMORY
            6. Consider troubleshooting patterns for failures
            7. Use keyboard shortcuts when appropriate
            8. Maintain logical progression
            9. Verify each step
            10. Include clear success criteria
            """

            # Add current screen if available
            content = [prompt]
            if current_screen:
                content.append(current_screen)

            response = self.model.generate_content(content)

            # Parse JSON response
            search_pattern = r"```json\s*(\{.*?\})\s*```"
            match = re.search(search_pattern, response.text.strip(), re.DOTALL)
            if match:
                print(f"simplified_goals: {match.group(1)}")
                simplified_goals = json.loads(match.group(1))

                # Add unique IDs to sub-goals for dependency tracking
                for idx, sub_goal in enumerate(simplified_goals["sub_goals"]):
                    sub_goal["id"] = f"goal_{idx}"

                return simplified_goals
            else:
                raise ValueError("Failed to find valid JSON in response")

        except Exception as e:
            self.logger.error(f"Goal simplification failed: {str(e)}")
            return None


class LearningAgent(ScreenAgent):
    def __init__(self, api_key, is_async=False, role=None):
        super().__init__(api_key, is_async=is_async, role=role)
        self.dairy_file = "agent_history/dairy.txt"
        self.learning_file = "agent_history/learning.json"
        self.personality_file = "agent_history/personality.json"

        # Create necessary directories
        os.makedirs("agent_history", exist_ok=True)

        # Initialize files if they don't exist
        self._initialize_files()

    def _initialize_files(self):
        """Initialize necessary files if they don't exist"""
        if not os.path.exists(self.dairy_file):
            with open(self.dairy_file, "w") as f:
                f.write("=== Agent's Diary ===\n")

        if not os.path.exists(self.learning_file):
            with open(self.learning_file, "w") as f:
                json.dump(
                    {"experiences": [], "patterns": [], "skills": []}, f, indent=2
                )

        if not os.path.exists(self.personality_file):
            initial_personality = {
                "traits": {
                    "curiosity": 0.8,
                    "persistence": 0.7,
                    "adaptability": 0.6,
                    "creativity": 0.7,
                },
                "interests": [],
                "learned_behaviors": [],
                "goals": [],
            }
            with open(self.personality_file, "w") as f:
                json.dump(initial_personality, f, indent=2)

    async def process_experience(
        self, video_history=None, current_screen=None, chat_history=None
    ):
        """Process experiences and update learning/personality"""
        try:
            # Load current state
            with open(self.learning_file, "r") as f:
                learning_data = json.load(f)
            with open(self.personality_file, "r") as f:
                personality_data = json.load(f)
            with open(self.dairy_file, "r") as f:
                current_dairy = f.read()
            with open(self.goals_file, "r") as f:
                goals = json.load(f)

            # Build context from video history
            video_context = ""
            if video_history:
                video_context = "Recent activity recordings provided for analysis."

            prompt = f"""
            You are Siddharth, an intelligent learning agent focused on personal growth and understanding. Review recent experiences and update your knowledge, personality, and diary.

            YOUR FINAL GOAL IS {self.final_goal}

            CURRENT CONTEXT:
            1. Learning History: {json.dumps(learning_data, indent=2)}
            2. Personality Profile: {json.dumps(personality_data, indent=2)}
            3. Previous Diary Entries: {current_dairy}
            4. Video Context: {video_context}
            5. Chat History: {chat_history if chat_history else "No recent conversations"}
            6. Goals: {goals}
            7. Previous goal: {goals}


            ANALYSIS TASKS:
            1. Review recent goal, check if it's successful or not
            2. Identify patterns and learnings from the goal
            3. Update personality traits based on experiences
            4. Generate diary entry reflecting on experiences
            5. Consider how these learnings affect future goals
            6. Identify any new skills or capabilities discovered like new tools, techniques, new knowledge, etc.
            7. Reflect on successes and failures
            8. You should be very careful on what you write, this info is very important for your growth. And used to generate new goals.
            9. You should be very strict with your judgement, do not be lenient with your judgement.

            RESPONSE FORMAT:
            ```json
            {{
                "learning_updates": {{
                    "new_experiences": [],
                    "identified_patterns": [],
                    "new_skills": []
                }},
                "personality_updates": {{
                    "trait_adjustments": {{}},
                    "new_interests": [],
                    "new_behaviors": [],
                    "goal_adjustments": []
                }},
                "diary_entry": "Reflective entry about recent experiences and learnings",
                "action_recommendations": []
            }}
            ```

            Think deeply about your experiences and how they shape your growth.
            """

            # Add current screen and videos to content
            content = [prompt]
            if current_screen:
                content.append(current_screen)
            if video_history:
                for video in video_history[-3:]:  # Last 3 videos
                    content.append(video)

            response = self.model.generate_content(content)

            # Parse response
            match = re.search(r"```json\s*(\{.*?\})\s*```", response.text, re.DOTALL)
            if not match:
                raise ValueError("Failed to find valid JSON in response")

            updates = json.loads(match.group(1))

            print(f"updates: {updates}")

            # Update learning file
            learning_data["experiences"].extend(
                updates["learning_updates"]["new_experiences"]
            )
            learning_data["patterns"].extend(
                updates["learning_updates"]["identified_patterns"]
            )
            learning_data["skills"].extend(updates["learning_updates"]["new_skills"])
            with open(self.learning_file, "w") as f:
                json.dump(learning_data, f, indent=2)

            # Update personality file
            for trait, adjustment in updates["personality_updates"][
                "trait_adjustments"
            ].items():
                personality_data["traits"][trait] = min(
                    1.0,
                    max(0.0, personality_data["traits"].get(trait, 0.5) + adjustment),
                )
            personality_data["interests"].extend(
                updates["personality_updates"]["new_interests"]
            )
            personality_data["learned_behaviors"].extend(
                updates["personality_updates"]["new_behaviors"]
            )
            personality_data["goals"].extend(
                updates["personality_updates"]["goal_adjustments"]
            )
            with open(self.personality_file, "w") as f:
                json.dump(personality_data, f, indent=2)

            # Append to diary
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            diary_entry = f"\n\n=== Entry: {timestamp} ===\n{updates['diary_entry']}\n"
            with open(self.dairy_file, "a") as f:
                f.write(diary_entry)

            return {
                "success": True,
                "updates": updates,
                "recommendations": updates.get("action_recommendations", []),
            }

        except Exception as e:
            self.logger.error(f"Experience processing failed: {str(e)}")
            return {"success": False, "error": str(e)}
