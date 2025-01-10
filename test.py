import google.generativeai as genai
import cv2
import numpy as np
import os
from datetime import datetime
import mss  # Add this import at the top
import json

# Add goal history context
goal_history = []
chat_history = []
screen_capture = mss.mss()  # For screen capture
fps = 10  # Frames per second of the recorded video
recording_duration = 2  # seconds

model = genai.GenerativeModel("gemini-2.0-flash-exp")


def load_chat_history():
    """Load chat history from file at startup"""
    try:
        if os.path.exists("agent_history/chat_history.json"):
            with open("agent_history/chat_history.json", "r") as f:
                chat_history = json.load(f)
        else:
            chat_history = []
    except Exception as e:
        print(f"Failed to load chat history: {e}")
        chat_history = []

    return chat_history


def record_screen():
    """Record screen for specified duration and return video path"""
    frames = []
    frame_count = fps * recording_duration

    # Use .avi format with XVID codec
    temp_video_path = os.path.join("screen_record_{datetime.now().timestamp()}.avi")
    screen = screen_capture.grab(screen_capture.monitors[1])  # Capture primary monitor
    height, width = screen.height, screen.width

    fourcc = cv2.VideoWriter.fourcc(*"XVID")
    out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))

    if not out.isOpened():
        raise Exception("Video writer initialization failed")

    for _ in range(frame_count):
        screen = screen_capture.grab(screen_capture.monitors[1])
        frame = np.array(screen)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        out.write(frame)

    out.release()
    return temp_video_path


def generate_goal(goal_history):
    """Goal Generator Role: Create next goal with chat history"""

    # try:
    goal_history_context = "\nPREVIOUS GOALS AND OUTCOMES:\n"
    for past_goal in goal_history:
        if isinstance(past_goal, dict):  # Check if entry is a dictionary
            goal_history_context += f"""
            Goal: {past_goal.get('goal', 'No goal')}
            Success: {past_goal.get('success', False)}
            Timestamp: {past_goal.get('timestamp', 'No timestamp')}
            Learning: {past_goal.get('learning', 'No learning recorded')}
            Personal Reflection: {past_goal.get('personal_reflection', 'No reflection')}
            Human Feedback: {past_goal.get('human_feedback', 'No feedback')}
            ---
            """
        else:
            goal_history_context += f"Entry: {past_goal}\n---\n"

    prompt = f"""
            You are Siddharth, a human being. Think like a human being. 
            Be conscious of your actions. Be conscious of your thoughts. Be conscious of your goals.
            Be consistently inconsistent.
            
            {goal_history_context}

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
            - Stop doing the same task again and again. Do it once and move on to the next task. Else you will be stuck in a loop. Which is bad.
        

            """

    # Build content list
    content = [prompt]

    # Add historical videos if available
    # if video_history:
    #     content.append("Here are the last few screen recordings for context:")
    #     for vid in video_history[-3:]:
    #         if vid.name != video_file.name:  # Only add if not the current video
    #             content.append(vid)

    # Add current video
    # content.append("Previous execution video:")
    # content.append(video_file)

    # Add current screen snapshot
    # if current_screen:
    #     print("Detected screen")
    #     content.append(
    #         "Current screen state(check this properly for better understanding):"
    #     )
    #     content.append(current_screen)

    content.append(
        "can you tell me what all you have learnt until this point based on the previous goals and outcomes, i want to every single thing you have learnt until this point, includng minute details"
    )

    # Debug log to see content structure
    print(f"Number of content items being sent: {len(content)}")
    print(
        f"Number of video files included: {len([x for x in content if hasattr(x, 'mime_type')])}"
    )

    response = model.generate_content(content)
    goal = response.text.strip()
    # print(goal)
    return goal


if __name__ == "__main__":
    goal_history: list[dict] = load_chat_history()
    # video_file = record_screen()
    # gemini_file = genai.upload_file(video_file, mime_type="video/avi")
    goal = generate_goal(goal_history)
    print(goal)
