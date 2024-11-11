import json
import logging
import tempfile

import anthropic
import openai


import traceback
from enum import Enum

import markdown
from dotenv import load_dotenv
from elevenlabs import ElevenLabs, Voice
from elevenlabs.core import RequestOptions
from fasthtml.common import *
from dataclasses import dataclass
import sqlite3
import os
from datetime import datetime, timedelta
from starlette.datastructures import UploadFile
from audio import transcribe_audio
from openai import OpenAI

from emder_tool import emdr_tool
from healing_scripts.get_default_example_healing_script import get_default_healing_script
from elevenlabs import play, save

from scripts.audio_recording_js import get_audio_recording_js

elevenlabs_api_key = os.environ.get('ELEVENLABS_API_KEY')

# Set your ElevenLabs API key
elevenlabs_client = ElevenLabs(
    api_key=elevenlabs_api_key  # Replace with your ElevenLabs API key
)

load_dotenv()

openai.api_key = os.environ.get('OPENAI_API_KEY')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
anthropic_client = anthropic.Client(api_key=os.getenv("ANTHROPIC_API_KEY"))


# Define the JournalEntry dataclass
@dataclass
class JournalEntry:
    id: int
    title: str
    content: str
    date: str


@dataclass
class Transformation:
    id: int
    description: str
    date_added: str

@dataclass
class Goalslide:
    id: int
    title: str
    description: str
    generated_visualization_script: str
    date: str
    visualization_audio: Optional[str] = None  # New field
@dataclass
class Challenge:
    id: int
    title: str
    description: str
    start_date: str
    end_date: str
    duration: int
    reminders_enabled: int


@dataclass
class ChallengeProgress:
    id: int
    challenge_id: int
    progress_date: str
    completed: int


@dataclass
class CBTEntry:
    id: int
    negative_thought: str
    cognitive_distortion: str
    rational_rebuttal: str
    date: str


@dataclass
class DecisionRecord:
    id: int
    title: str
    content: str
    date: str


@dataclass
class NoteEntry:
    id: int
    title: str
    content: str
    date: str


# Define the MoodEntry dataclass
@dataclass
class MoodEntry:
    id: int
    mood: str
    note: str
    date: str


# Initialize the app and router
app, rt = fast_app()

# Initialize SQLite database
conn = sqlite3.connect('journal2.db', check_same_thread=False)
cursor = conn.cursor()


# Create transformations table
cursor.execute('''
CREATE TABLE IF NOT EXISTS transformations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    description TEXT NOT NULL,
    date_added TEXT NOT NULL
)
''')
conn.commit()

@dataclass
class EMDRSession:
    id: int
    feelings: str
    memory: str
    body_sensations: str
    negative_beliefs: str
    positive_beliefs: str
    date: str
    session_name: str
    healing_script: Optional[str] = None  # New field
    healing_script_audio: Optional[str] = None  # New field
    target_description: Optional[str] = None  # New field
    integration_script: Optional[str] = None  # New field
    integration_script_audio: Optional[str] = None  # New field
    reprogramming_script: Optional[str] = None
    reprogramming_script_audio: Optional[str] = None


# Create emdr_sessions table
cursor.execute('''
CREATE TABLE IF NOT EXISTS emdr_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    feelings TEXT NOT NULL,
    memory TEXT NOT NULL,
    body_sensations TEXT NOT NULL,
    negative_beliefs TEXT NOT NULL,
    positive_beliefs TEXT NOT NULL,
    date TEXT NOT NULL
)
''')
try:
    cursor.execute('ALTER TABLE emdr_sessions ADD COLUMN session_name TEXT')
    conn.commit()
except sqlite3.OperationalError as e:
    if 'duplicate column name' in str(e):
        pass  # Column already exists
    else:
        raise e

try:
    cursor.execute('ALTER TABLE emdr_sessions ADD COLUMN healing_script TEXT')
    conn.commit()
except sqlite3.OperationalError as e:
    if 'duplicate column name' in str(e):
        pass  # Column already exists
    else:
        raise e

try:
    cursor.execute('ALTER TABLE emdr_sessions ADD COLUMN healing_script_audio TEXT')
    conn.commit()
except sqlite3.OperationalError as e:
    if 'duplicate column name' in str(e):
        pass  # Column already exists
    else:
        raise e

try:
    cursor.execute('ALTER TABLE emdr_sessions ADD COLUMN integration_script TEXT')
    conn.commit()
except sqlite3.OperationalError as e:
    if 'duplicate column name' in str(e):
        pass  # Column already exists
    else:
        raise e

try:
    cursor.execute('ALTER TABLE emdr_sessions ADD COLUMN integration_script_audio TEXT')
    conn.commit()
except sqlite3.OperationalError as e:
    if 'duplicate column name' in str(e):
        pass  # Column already exists
    else:
        raise e

cursor.execute('''
    UPDATE emdr_sessions
    SET integration_script = '', integration_script_audio = ''
    WHERE integration_script IS NULL OR integration_script_audio IS NULL
''')
conn.commit()

# After creating the emdr_sessions table and adding other columns
try:
    cursor.execute('ALTER TABLE emdr_sessions ADD COLUMN reprogramming_script TEXT')
    conn.commit()
except sqlite3.OperationalError as e:
    if 'duplicate column name' in str(e):
        pass  # Column already exists
    else:
        raise e

try:
    cursor.execute('ALTER TABLE emdr_sessions ADD COLUMN reprogramming_script_audio TEXT')
    conn.commit()
except sqlite3.OperationalError as e:
    if 'duplicate column name' in str(e):
        pass
    else:
        raise e



try:
    cursor.execute('ALTER TABLE emdr_sessions ADD COLUMN target_description TEXT')
    conn.commit()
except sqlite3.OperationalError as e:
    if 'duplicate column name' in str(e):
        pass  # Column already exists
    else:
        raise e

cursor.execute('''
    UPDATE emdr_sessions
    SET reprogramming_script = '', reprogramming_script_audio = ''
    WHERE reprogramming_script IS NULL OR reprogramming_script_audio IS NULL
''')
conn.commit()
# Create journal_entries table if it doesn't exist
cursor.execute('''
CREATE TABLE IF NOT EXISTS journal_entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    date TEXT NOT NULL
)
''')

# Create goalslides table
cursor.execute('''
CREATE TABLE IF NOT EXISTS goalslides (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    generated_visualization_script TEXT,
    date TEXT NOT NULL
)
''')

# Create mood_entries table if it doesn't exist
cursor.execute('''
CREATE TABLE IF NOT EXISTS mood_entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    mood TEXT NOT NULL,
    note TEXT,
    date TEXT NOT NULL
)
''')

# Create cbt_entries table
cursor.execute('''
CREATE TABLE IF NOT EXISTS cbt_entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    negative_thought TEXT NOT NULL,
    cognitive_distortion TEXT NOT NULL,
    rational_rebuttal TEXT NOT NULL,
    date TEXT NOT NULL
)
''')

# Create weekly_plans table
cursor.execute('''
CREATE TABLE IF NOT EXISTS weekly_plans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    plan TEXT NOT NULL
)
''')
conn.commit()

cursor.execute('''
CREATE TABLE IF NOT EXISTS tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    description TEXT NOT NULL,
    priority INTEGER DEFAULT 4,
    status TEXT DEFAULT 'pending'
)
''')

try:
    cursor.execute('ALTER TABLE tasks ADD COLUMN category TEXT')
    conn.commit()
except sqlite3.OperationalError as e:
    if 'duplicate column name' in str(e):
        pass  # Column already exists
    else:
        raise e

try:
    cursor.execute('ALTER TABLE tasks ADD COLUMN urgency INTEGER DEFAULT 5')
    conn.commit()
except sqlite3.OperationalError as e:
    if 'duplicate column name' in str(e):
        pass  # Column already exists
    else:
        raise e

# Create notes_entries table if it doesn't exist
cursor.execute('''
CREATE TABLE IF NOT EXISTS notes_entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    date TEXT NOT NULL
)
''')
conn.commit()

cursor.execute('''
CREATE TABLE IF NOT EXISTS daily_schedules (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    schedule TEXT NOT NULL
)
''')
conn.commit()


@dataclass
class ChallengeIdea:
    id: int
    title: str
    description: str
    date_added: str


@dataclass
class EMDRTarget:
    id: int
    content: str
    date_generated: str


# Create emdr_targets table
cursor.execute('''
CREATE TABLE IF NOT EXISTS emdr_targets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content TEXT NOT NULL,
    date_generated TEXT NOT NULL
)
''')
conn.commit()

# Create challenge_ideas table
cursor.execute('''
CREATE TABLE IF NOT EXISTS challenge_ideas (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    date_added TEXT NOT NULL
)
''')
conn.commit()

# Create challenges table
cursor.execute('''
CREATE TABLE IF NOT EXISTS challenges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    start_date TEXT NOT NULL,
    end_date TEXT NOT NULL,
    reminders_enabled INTEGER DEFAULT 0
)
''')

# Create challenge_progress table
cursor.execute('''
CREATE TABLE IF NOT EXISTS challenge_progress (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    challenge_id INTEGER NOT NULL,
    progress_date TEXT NOT NULL,
    completed INTEGER DEFAULT 0,
    FOREIGN KEY(challenge_id) REFERENCES challenges(id)
)
''')

try:
    cursor.execute('ALTER TABLE challenges ADD COLUMN duration INTEGER DEFAULT 30')
    conn.commit()
except sqlite3.OperationalError:
    # Column already exists
    pass

# Create decision_records table
cursor.execute('''
CREATE TABLE IF NOT EXISTS decision_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    date TEXT NOT NULL
)
''')

conn.commit()


class Priority(Enum):
    HIGH = 1
    MEDIUM = 2
    LOW = 3
    NONE = 4


def parse_task_input(input_string):
    priority_map = {'p1': Priority.HIGH, 'p2': Priority.MEDIUM, 'p3': Priority.LOW}
    priority_match = re.search(r'\s(p[1-3])(\s|$)', input_string)
    priority = priority_map[priority_match.group(1)] if priority_match else Priority.NONE
    description = re.sub(r'\s(p[1-3])(\s|$)', ' ', input_string).strip()
    return description, priority.value


# Helper function to find a journal entry by ID
def find_journal_entry(entry_id):
    cursor.execute('SELECT * FROM journal_entries WHERE id = ?', (entry_id,))
    row = cursor.fetchone()
    if row:
        return JournalEntry(id=row[0], title=row[1], content=row[2], date=row[3])
    return None


# Helper function to find a mood entry by ID
def find_mood_entry(entry_id):
    cursor.execute('SELECT * FROM mood_entries WHERE id = ?', (entry_id,))
    row = cursor.fetchone()
    if row:
        return MoodEntry(id=row[0], mood=row[1], note=row[2], date=row[3])
    return None

def categorize_task(description):
    # Get existing categories
    cursor.execute('SELECT DISTINCT category FROM tasks')
    existing_categories = [row[0] for row in cursor.fetchall() if row[0]]

    existing_categories_text = ', '.join(existing_categories) if existing_categories else 'None'

    # Construct messages for ChatGPT
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that categorizes tasks and determines their urgency on a scale of 1 to 10."
        },
        {
            "role": "user",
            "content": f"""
Please categorize the following task and determine its urgency on a scale from 1 (least urgent) to 10 (most urgent). Use the existing categories or create a new one if necessary.

Task description: {description}

Existing categories: {existing_categories_text}
"""
        }
    ]

    # Define the function schema
    functions = [
        {
            "name": "categorize_task",
            "description": "Categorizes a task and determines its urgency",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "Category that best fits the task; use an existing category or suggest a new one"
                    },
                    "urgency": {
                        "type": "integer",
                        "description": "Urgency of the task on a scale from 1 (least urgent) to 10 (most urgent)"
                    },
                },
                "required": ["category", "urgency"],
            },
        },
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Use the appropriate model
            messages=messages,
            functions=functions,
            function_call={"name": "categorize_task"},
        )

        message = response.choices[0].message

        if message.function_call:
            function_args = json.loads(message.function_call.arguments)
            category = function_args.get("category")
            urgency = int(function_args.get("urgency"))

            return category, urgency

        else:
            return "General", 5  # Default values if ChatGPT doesn't return values

    except Exception as e:
        print(f"Error categorizing task: {e}")
        return "General", 5  # Default values on error


def create_tasks_subtabs(active_subtab):
    tasks_subtabs = [
        ("Inbox", "/tasks"),
        ("Daily Schedules", "/tasks/schedules"),
        ("Weekly Plan", "/tasks/weekly"),
    ]

    subtab_elements = [
        A(
            name,
            href=url,
            cls=f"subtab {'active' if name.lower() == active_subtab.lower() else ''}"
        )
        for name, url in tasks_subtabs
    ]

    return Div(*subtab_elements, cls="subtab-navigation")


def render_decision_record(entry):
    return Div(
        H3(entry.title),
        Div(NotStr(markdown_to_html(entry.content)), cls="markdown-content"),
        P(f"Date: {entry.date}", cls="entry-date"),
        Div(
            A("Edit", href=f"/decision-record/edit/{entry.id}", cls="edit-link"),
            A("Delete", hx_delete=f"/decision-record/delete/{entry.id}", hx_swap="outerHTML", cls="delete-link"),
            cls="entry-actions"
        ),
        cls="decision-record-entry",
        id=f"decision-record-{entry.id}"
    )


def find_note_entry(entry_id):
    cursor.execute('SELECT * FROM notes_entries WHERE id = ?', (entry_id,))
    row = cursor.fetchone()
    if row:
        return NoteEntry(id=row[0], title=row[1], content=row[2], date=row[3])
    return None


def render_note_entry(entry):
    return Div(
        H3(entry.title),
        P(entry.content),
        P(f"Date: {entry.date}", cls="entry-date"),
        Div(
            A("Edit", href=f"/notes/edit/{entry.id}", cls="edit-link"),
            A("Delete", hx_delete=f"/notes/delete/{entry.id}", hx_swap="outerHTML", cls="delete-link"),
            Button(
                "Auto-Title",
                hx_post=f"/notes/auto-title/{entry.id}",
                hx_target=f"#note-entry-{entry.id}",
                hx_swap="outerHTML",
                cls="auto-title-button"
            ),
            cls="entry-actions"
        ),
        cls="note-entry",
        id=f"note-entry-{entry.id}"
    )


def markdown_to_html(text):
    return markdown.markdown(text)


# Helper function to render a single journal entry
def render_journal_entry(entry):
    return Div(
        H3(entry.title),
        P(entry.content),
        P(f"Date: {entry.date}", cls="entry-date"),
        Div(
            A("Edit", href=f"/journal/edit/{entry.id}", cls="edit-link"),
            A("Delete", hx_delete=f"/journal/delete/{entry.id}", hx_swap="outerHTML", cls="delete-link"),
            Button("Auto-Title", hx_post=f"/journal/auto-title/{entry.id}", hx_target=f"#entry-{entry.id}",
                   hx_swap="outerHTML", cls="auto-title-button"),
            Button("Generate CBT Entries", hx_post=f"/journal/generate-cbt/{entry.id}", cls="generate-cbt-button"),
            Button("Generate Tasks", hx_post=f"/journal/generate-tasks/{entry.id}", cls="generate-tasks-button"),
            cls="entry-actions"
        ),
        cls="journal-entry",
        id=f"entry-{entry.id}"
    )


def render_cbt_entry(entry):
    return Div(
        H3(f"Date: {entry.date}"),
        P(f"Negative Thought: {entry.negative_thought}"),
        P(f"Cognitive Distortion: {entry.cognitive_distortion}"),
        P(f"Rational Rebuttal: {entry.rational_rebuttal}"),
        Div(
            A("Edit", href=f"/cbt/edit/{entry.id}", cls="edit-link"),
            A("Delete", hx_delete=f"/cbt/delete/{entry.id}", hx_swap="outerHTML", cls="delete-link"),
            cls="entry-actions"
        ),
        cls="cbt-entry",
        id=f"cbt-entry-{entry.id}"
    )


# Helper function to render a single mood entry
def render_mood_entry(entry):
    return Div(
        H3(f"Mood: {entry.mood}"),
        P(entry.note) if entry.note else "",
        P(f"Date: {entry.date}", cls="entry-date"),
        Div(
            A("Edit", href=f"/mood/edit/{entry.id}", cls="edit-link"),
            A("Delete", hx_delete=f"/mood/delete/{entry.id}", hx_swap="outerHTML", cls="delete-link"),
            cls="entry-actions"
        ),
        cls="mood-entry",
        id=f"mood-entry-{entry.id}"
    )


# Helper function to create tab navigation
def create_tabs(active_tab):
    tabs = [
        ("Journal", "/"),
        ("Notes", "/notes"),
        ("Mood Log", "/mood"),
        ("Tasks", "/tasks"),
        ("CBT", "/cbt"),
        ("Decision Record", "/decision-record"),
        ("EMDR", "/emdr"),
        ("Goalslide", "/goalslide"),
        ("Challenges", "/challenges"),  # Added "Challenges" tab
    ]
    return Div(
        Img(src="/static/logo.png", alt="Journal Logo", cls="journal-logo"),
        *(A(name, href=url, cls=f"tab {'active' if name.lower() == active_tab.lower() else ''}") for name, url in tabs),
        cls="tab-navigation"
    )


@rt("/notes/auto-title/{entry_id}", methods=["POST"])
def auto_title_notes(entry_id: int):
    entry = find_note_entry(entry_id)
    if entry:
        content = entry.content
        # Construct the prompt for the OpenAI API
        prompt = f"Provide a concise and informative title for the following note content:\n\n{content}\n\nTitle:"
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # or "gpt-4" if you have access
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            new_title = response.choices[0].message.content.strip().replace('"', '')
            # Update the note's title in the database
            cursor.execute('UPDATE notes_entries SET title = ? WHERE id = ?', (new_title, entry_id))
            conn.commit()
            # Fetch the updated entry
            updated_entry = find_note_entry(entry_id)
            # Return the updated note entry
            return render_note_entry(updated_entry)
        except Exception as e:
            # Handle API errors
            return Div(f"Error generating title: {str(e)}", cls="error-message")
    else:
        return Div("Note entry not found.", cls="error-message")


@rt("/notes", methods=["GET"])
def get_notes():
    cursor.execute('SELECT * FROM notes_entries ORDER BY date DESC')
    entries = [render_note_entry(NoteEntry(id=row[0], title=row[1], content=row[2], date=row[3]))
               for row in cursor.fetchall()]

    add_form = Form(
        H3("Add New Note"),
        Input(name="title", placeholder="Title"),
        Textarea(name="content", placeholder="Content"),
        Button("Add", cls="add-button"),
        hx_post="/notes/add",
        hx_swap="afterbegin",
        hx_target="#notes-entries-list",
        cls="add-entry-form"
    )

    # Voice recording buttons and status
    record_button = Button("Start Voice Recording", id="start-recording-button", cls="voice-record-button")
    stop_button = Button("Stop Voice Recording", id="stop-recording-button", style="display:none;",
                         cls="voice-record-button")
    status_div = Div(id="recording-status", cls="recording-status")

    layout = Div(
        create_tabs("Notes"),
        Div(
            Div(add_form, record_button, stop_button, status_div, cls="sidebar"),
            Div(Div(*entries, id="notes-entries-list"), cls="main-content"),
            cls="content-layout"
        ),
        cls="layout"
    )

    # Include the JavaScript with the same 'audio_recording_js' variable
    return layout, Script(audio_recording_js), Link(rel="stylesheet", href="/static/styles.css")


@rt("/notes/add", methods=["POST"])
def post_note(title: str, content: str):
    date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('INSERT INTO notes_entries (title, content, date) VALUES (?, ?, ?)', (title, content, date))
    conn.commit()
    new_entry_id = cursor.lastrowid
    new_entry = find_note_entry(new_entry_id)
    return render_note_entry(new_entry)


@rt("/notes/delete/{entry_id}", methods=["DELETE"])
def delete_note(entry_id: int):
    cursor.execute('DELETE FROM notes_entries WHERE id = ?', (entry_id,))
    conn.commit()
    return ""


@rt("/notes/edit/{entry_id}", methods=["GET"])
def get_note_edit(entry_id: int):
    entry = find_note_entry(entry_id)
    if entry:
        edit_form = Form(
            Input(name="title", value=entry.title),
            Textarea(name="content", value=entry.content),
            Button("Save", cls="save-button"),
            hx_put=f"/notes/edit/{entry.id}",
            hx_swap="outerHTML",
            cls="edit-entry-form"
        )
        return Div(edit_form, id=f"note-entry-{entry.id}")


@rt("/notes/edit/{entry_id}", methods=["PUT"])
def put_note(entry_id: int, title: str, content: str):
    cursor.execute('UPDATE notes_entries SET title = ?, content = ? WHERE id = ?', (title, content, entry_id))
    conn.commit()
    updated_entry = find_note_entry(entry_id)
    return render_note_entry(updated_entry)


@rt("/emdr/tool", methods=["GET"])
def get_emdr_tool():
    return emdr_tool()


@rt("/tasks/weekly", methods=["GET"])
def get_weekly_plan():
    # Fetch the latest weekly plan
    cursor.execute('SELECT * FROM weekly_plans ORDER BY date DESC LIMIT 1')
    weekly_plan = cursor.fetchone()

    if weekly_plan:
        plan_entry = render_weekly_plan_entry(weekly_plan)
    else:
        plan_entry = Div("No weekly plan generated yet.", cls="no-weekly-plan")

    generate_plan_button = Button("Generate Weekly Plan", hx_post="/generate-weekly-plan",
                                  cls="generate-weekly-plan-button")

    layout = Div(
        create_tabs("Tasks"),
        create_tasks_subtabs("Weekly Plan"),
        Div(
            generate_plan_button,
            plan_entry,
            cls="weekly-plan-container"
        ),
        cls="layout"
    )

    return layout, Link(rel="stylesheet", href="/static/styles.css")


@rt("/generate-weekly-plan", methods=["POST"])
def generate_weekly_plan():
    try:
        # Retrieve all pending tasks
        cursor.execute('SELECT description FROM tasks WHERE status = "pending" ORDER BY priority ASC')
        tasks = [desc for (desc,) in cursor.fetchall()]

        # Retrieve the latest journal entries from the past week
        cursor.execute('SELECT content FROM journal_entries WHERE date >= date("now", "-7 days")')
        journal_entries = [content for (content,) in cursor.fetchall()]

        # Prepare data for OpenAI API
        tasks_text = "\n".join([f"- {desc}" for desc in tasks])
        journal_text = "\n\n".join(journal_entries)

        # Construct the prompt
        prompt = f"""
Based on the following tasks and journal entries, create a weekly plan focusing on 1-3 major things to focus on this week. Also, provide a suggested schedule from Monday to Sunday.

Tasks:
{tasks_text}

Journal Entries:
{journal_text}

Please provide the plan in the following format:

Major Focus Areas:
1. [First major focus]
2. [Second major focus]
3. [Third major focus]

Weekly Schedule:
Monday: [Planned activities]
Tuesday: [Planned activities]
...
Sunday: [Planned activities]
"""

        # Call the OpenAI API
        response = client.chat.completions.create(
            model="o1-preview",  # Use "gpt-4" or "gpt-3.5-turbo" depending on your access
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        plan_text = response.choices[0].message.content

        # Save the plan into the database
        date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute('INSERT INTO weekly_plans (date, plan) VALUES (?, ?)', (date_str, plan_text))
        conn.commit()

        # Redirect to the weekly plan page
        return RedirectResponse("/tasks/weekly", status_code=303)

    except Exception as e:
        return Div(f"Error generating weekly plan: {str(e)}", cls="error-message")


@rt("/tasks/weekly/delete/{plan_id}", methods=["DELETE"])
def delete_weekly_plan(plan_id: int):
    cursor.execute('DELETE FROM weekly_plans WHERE id = ?', (plan_id,))
    conn.commit()
    return ""


def render_weekly_plan_entry(plan_entry):
    plan_id, date, plan_text = plan_entry
    return Div(
        H3(f"Weekly Plan for {date.split(' ')[0]}"),
        Pre(plan_text, cls="weekly-plan-text"),
        Div(
            A("Delete", hx_delete=f"/tasks/weekly/delete/{plan_id}", hx_swap="outerHTML", cls="delete-link"),
            cls="entry-actions"
        ),
        cls="weekly-plan-entry",
        id=f"weekly-plan-{plan_id}"
    )


def find_entry(entry_id):
    cursor.execute('SELECT * FROM journal_entries WHERE id = ?', (entry_id,))
    row = cursor.fetchone()
    if row:
        return JournalEntry(id=row[0], title=row[1], content=row[2], date=row[3])
    return None


# Helper function to render a single journal entry
def render_entry(entry):
    return Div(
        H3(entry.title),
        P(entry.content),
        P(f"Date: {entry.date}", cls="entry-date"),
        Div(
            A("Edit", href=f"/edit/{entry.id}", cls="edit-link"),
            A("Delete", hx_delete=f"/delete/{entry.id}", hx_swap="outerHTML", cls="delete-link"),
            cls="entry-actions"
        ),
        cls="journal-entry",
        id=f"entry-{entry.id}"
    )


def generate_beliefs_claude(prompt: str):
    """Call Claude API to generate beliefs based on the prompt."""
    try:
        response = anthropic_client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        print(response.content)
        return response.content
    except Exception as e:
        return f"Error calling Claude API: {str(e)}"


def process_beliefs_with_gpt(beliefs_text: str, belief_type: str):
    """Use GPT function calling to process beliefs into a list."""
    messages = [
        {"role": "system", "content": "You are an assistant that formats beliefs into structured lists."},
        {"role": "user",
         "content": f"Please turn the following text into an array of {belief_type} beliefs:\n{beliefs_text}"}
    ]

    functions = [
        {
            "name": "extract_beliefs",
            "description": f"Extracts a list of {belief_type} beliefs from text",
            "parameters": {
                "type": "object",
                "properties": {
                    "beliefs": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "description": f"A {belief_type} belief from the text"
                        }
                    }
                },
                "required": ["beliefs"],
            }
        }
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # Or "gpt-3.5-turbo"
            messages=messages,
            functions=functions,
            function_call={"name": "extract_beliefs"},
        )
        function_call_result = response.choices[0].message.function_call.arguments
        beliefs_list = json.loads(function_call_result).get("beliefs", [])
        return beliefs_list
    except Exception as e:
        return f"Error processing beliefs with GPT: {str(e)}"


def render_emdr_session(session):
    # Parse the stored negative and positive beliefs into lists
    negative_beliefs_list = session.negative_beliefs.strip().split(', ') if session.negative_beliefs else []
    positive_beliefs_list = session.positive_beliefs.strip().split(', ') if session.positive_beliefs else []

    try:
        # Healing Script Section
        if session.healing_script:
            # Include the audio player if the audio file exists
            if session.healing_script_audio:
                audio_player = Audio(
                    Source(src=f"/static/audio/{session.healing_script_audio}", type="audio/mpeg"),
                    controls=True,
                    cls="healing-script-audio"
                )
            else:
                # Audio not available; provide a button to generate it
                audio_player = Div(
                    Div("Audio not available.", cls="audio-not-available"),
                    Button(
                        "Generate Healing Script Audio",
                        hx_post=f"/emdr/generate-healing-audio/{session.id}",
                        hx_target=f"#healing-script-{session.id}",
                        hx_swap="outerHTML",
                        cls="generate-healing-audio-button"
                    ),
                    cls="healing-script-audio-generator"
                )
            healing_script_section = Div(
                Summary("Healing Script"), Br(),
                Div(
                    Pre(session.healing_script),
                    audio_player,
                    cls="healing-script-content"
                ), Br(),
                id=f"healing-script-{session.id}",
                cls="healing-script-section"
            )
        else:
            healing_script_section = Div(
                Button(
                    "Generate Healing Script",
                    hx_post=f"/emdr/generate-healing-script/{session.id}",
                    hx_target=f"#healing-script-{session.id}",
                    hx_swap="outerHTML",
                    cls="generate-healing-script-button"
                ),
                id=f"healing-script-{session.id}"
            )
    except Exception as e:
        print(f"Error in healing_script_section for session {session.id}: {e}")
        traceback.print_exc()
        healing_script_section = Div(f"Error rendering healing script: {str(e)}", cls="error-message")

    if hasattr(session, 'integration_script'):
        if session.integration_script:
            # Include the audio player if the audio file exists
            if session.integration_script_audio:
                audio_player_integration = Audio(
                    Source(src=f"/static/audio/{session.integration_script_audio}", type="audio/mpeg"),
                    controls=True,
                    cls="integration-script-audio"
                )
            else:
                # Audio not available; provide a button to generate it
                audio_player_integration = Div(
                    Div("Audio not available.", cls="audio-not-available"),
                    Button(
                        "Generate Integration Script Audio",
                        hx_post=f"/emdr/generate-integration-audio/{session.id}",
                        hx_target=f"#integration-script-{session.id}",
                        hx_swap="outerHTML",
                        cls="generate-integration-audio-button"
                    ),
                    cls="integration-script-audio-generator"
                )

            integration_script_section = Div(
                Summary("Integration Script"), Br(),
                Div(
                    Pre(session.integration_script),
                    Br(),
                    audio_player_integration,
                    cls="integration-script-content"
                ),
                id=f"integration-script-{session.id}",
                cls="integration-script-section"
            )
        else:
            integration_script_section = Div(
                Button(
                    "Generate Integration Script",
                    hx_post=f"/emdr/generate-integration-script/{session.id}",
                    hx_target=f"#integration-script-{session.id}",
                    hx_swap="outerHTML",
                    cls="generate-integration-script-button"
                ),
                id=f"integration-script-{session.id}"
            )
    else:
        # Handle the case where the session does not have the integration_script attribute
        integration_script_section = Div(
            "Integration Script feature not available for this session.",
            cls="integration-script-unavailable"
        )

                # Now proceed to render the rest of the session
    try:
        if session.reprogramming_script:
            if session.reprogramming_script_audio:
                audio_player_reprogramming = Audio(
                    Source(src=f"/static/audio/{session.reprogramming_script_audio}", type="audio/mpeg"),
                    controls=True,
                    cls="reprogramming-script-audio"
                )
            else:
                audio_player_reprogramming = Div(
                    Div("Audio not available.", cls="audio-not-available"),
                    Button(
                        "Generate Reprogramming Script Audio",
                        hx_post=f"/emdr/generate-reprogramming-audio/{session.id}",
                        hx_target=f"#reprogramming-script-{session.id}",
                        hx_swap="outerHTML",
                        cls="generate-reprogramming-audio-button"
                    ),
                    cls="reprogramming-script-audio-generator"
                )
            reprogramming_script_section = Div(
                Summary("Reprogramming Script"), Br(),
                Div(
                    Pre(session.reprogramming_script),
                    audio_player_reprogramming,
                    cls="reprogramming-script-content"
                ), Br(),
                id=f"reprogramming-script-{session.id}",
                cls="reprogramming-script-section"
            )
        else:
            reprogramming_script_section = Div(
                Button(
                    "Generate Reprogramming Script",
                    hx_post=f"/emdr/generate-reprogramming-script/{session.id}",
                    hx_target=f"#reprogramming-script-{session.id}",
                    hx_swap="outerHTML",
                    cls="generate-reprogramming-script-button"
                ),
                id=f"reprogramming-script-{session.id}"
            )
    except Exception as e:
        print(f"Error in reprogramming_script_section for session {session.id}: {e}")
        traceback.print_exc()
        reprogramming_script_section = Div(f"Error rendering reprogramming script: {str(e)}", cls="error-message")

        # Now include the reprogramming script section in the overall rendering
    return Details(
        Summary(session.session_name or f"Session on {session.date}"),
        Div(
            P(f"Feelings: {session.feelings}"),
            P(f"Memory: {session.memory}"),
            P(f"Body Sensations: {session.body_sensations}"),
            P(f"Target Description: {session.target_description or 'N/A'}"),
            # Negative Beliefs Section
            Div(
                Div(
                    Button(
                        "Identify Negative Beliefs",
                        hx_post=f"/emdr/generate-beliefs/{session.id}/negative",
                        hx_target=f"#negative-beliefs-{session.id}",
                        hx_swap="outerHTML",
                        cls="generate-button"
                    ),
                    Button(
                        "Develop Positive Beliefs",
                        hx_post=f"/emdr/generate-beliefs/{session.id}/positive",
                        hx_target=f"#positive-beliefs-{session.id}",
                        hx_swap="outerHTML",
                        cls="generate-button"
                    ),
                    cls="button-group"
                ),
                cls="session-actions"
            ),
            Div(
                Div(
                    H3("Negative Beliefs"),
                    Ul(*(Li(belief) for belief in negative_beliefs_list)),
                    Button(
                        "Play Negative Beliefs",
                        onclick=f"playBeliefs({session.id}, 'negative')",
                        cls="play-beliefs-button"
                    ),
                    id=f"negative-beliefs-{session.id}",
                    cls="beliefs-list"
                ),
                cls="beliefs-section"
            ),
            # Positive Beliefs Section
            Div(
                Div(
                    H3("Positive Beliefs"),
                    Ul(*(Li(belief) for belief in positive_beliefs_list)),
                    Button(
                        "Play Positive Beliefs",
                        onclick=f"playBeliefs({session.id}, 'positive')",
                        cls="play-beliefs-button"
                    ),
                    id=f"positive-beliefs-{session.id}",
                    cls="beliefs-list"
                ),
                cls="beliefs-section"
            ),
            # Entry Actions
            healing_script_section,
            integration_script_section,
            reprogramming_script_section,  # Include the new section here
            Br(),
            Div(
                A("Edit", href=f"/emdr/edit/{session.id}", cls="edit-link"),
                A("Delete", hx_delete=f"/emdr/delete/{session.id}", hx_swap="outerHTML", cls="delete-link"),
                Button(
                    "Auto-Title",
                    hx_post=f"/emdr/auto-title/{session.id}",
                    hx_target=f"#emdr-session-{session.id} summary",
                    hx_swap="outerHTML",
                    cls="auto-title-button"
                ),
                cls="entry-actions"
            ),
            cls="emdr-session-content"
        ), Hr(),
        cls="emdr-session-entry",
        id=f"emdr-session-{session.id}"
    )

@rt("/emdr/generate-reprogramming-script/{session_id}", methods=["POST"])
def generate_reprogramming_script(session_id: int):
    # Fetch the EMDR session
    cursor.execute('SELECT * FROM emdr_sessions WHERE id = ?', (session_id,))
    row = cursor.fetchone()
    if not row:
        return Div("Session not found.", cls="error-message")

    session = EMDRSession(*row)

    # Collect all the info associated with the EMDR session
    emdr_info = f"""
Feelings: {session.feelings}
Memory: {session.memory}
Body Sensations: {session.body_sensations}
Target Description: {session.target_description}
Negative Beliefs: {session.negative_beliefs}
Positive Beliefs: {session.positive_beliefs}
"""

    # Construct the prompt for the Claude client
    prompt = f"""
Can you please create a 'reprogramming script' that is so densely packed with metaphor, symbolism, imagery, emotions, jokes, etc., that it's too much data for my brain to process on the first time, and all of these things are aimed at integrating the positive beliefs?

EMDR Session Details:
{emdr_info}
"""

    # Call the Claude API
    try:
       response = anthropic_client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}]
        )
       reprogramming_script = ''.join([block.text for block in response.content if block.type == 'text'])

       # Save the reprogramming script to the database
       cursor.execute('UPDATE emdr_sessions SET reprogramming_script = ? WHERE id = ?', (reprogramming_script, session_id))
       conn.commit()

        # Generate audio using ElevenLabs
       audio = elevenlabs_client.generate(
           text=reprogramming_script,
           voice=Voice(
               voice_id="MiueK1FXuZTCItgbQwPu",
               name="Maya - Young Australian Female",
           ),
           request_options=RequestOptions(timeout_in_seconds=3000)
       )

       # Save the audio file
       audio_filename = f"reprogramming_script_{session_id}.mp3"
       audio_file_path = os.path.join('static', 'audio', audio_filename)
       save(audio, audio_file_path)

       # Update the database with the audio file path
       cursor.execute('UPDATE emdr_sessions SET reprogramming_script_audio = ? WHERE id = ?', (audio_filename, session_id))
       conn.commit()

       # Return the updated reprogramming script section
       # Re-fetch the updated session
       cursor.execute('SELECT * FROM emdr_sessions WHERE id = ?', (session_id,))
       updated_row = cursor.fetchone()
       updated_session = EMDRSession(*updated_row)
       return render_reprogramming_script_section(updated_session)

    except Exception as e:
        return Div(f"Error generating reprogramming script: {str(e)}", cls="error-message")

@rt("/emdr/generate-reprogramming-audio/{session_id}", methods=["POST"])
def generate_reprogramming_audio(session_id: int):
    # Fetch the EMDR session
    cursor.execute('SELECT * FROM emdr_sessions WHERE id = ?', (session_id,))
    row = cursor.fetchone()
    if not row:
        return Div("Session not found.", cls="error-message")

    session = EMDRSession(*row)

    if not session.reprogramming_script:
        return Div("Reprogramming script text not found. Please generate the reprogramming script first.", cls="error-message")

    # Generate audio using ElevenLabs
    try:
        audio = elevenlabs_client.generate(
            text=session.reprogramming_script,
            voice=Voice(
                voice_id="MiueK1FXuZTCItgbQwPu",
                name="Maya - Young Australian Female",
            ),
            request_options=RequestOptions(timeout_in_seconds=3000)
        )

        # Ensure the 'static/audio' directory exists
        os.makedirs(os.path.join('static', 'audio'), exist_ok=True)

        # Save the audio file
        audio_filename = f"reprogramming_script_{session_id}.mp3"
        audio_file_path = os.path.join('static', 'audio', audio_filename)
        save(audio, audio_file_path)

        # Update the database with the audio file path
        cursor.execute('UPDATE emdr_sessions SET reprogramming_script_audio = ? WHERE id = ?', (audio_filename, session_id))
        conn.commit()

        # Re-fetch the updated session
        cursor.execute('SELECT * FROM emdr_sessions WHERE id = ?', (session_id,))
        updated_row = cursor.fetchone()
        updated_session = EMDRSession(*updated_row)

        # Re-render the reprogramming script section
        return render_reprogramming_script_section(updated_session)

    except Exception as e:
        return Div(f"Error generating reprogramming script audio: {str(e)}", cls="error-message")

def render_reprogramming_script_section(session):
    if session.reprogramming_script:
        if session.reprogramming_script_audio:
            audio_player = Audio(
                Source(src=f"/static/audio/{session.reprogramming_script_audio}", type="audio/mpeg"),
                controls=True,
                cls="reprogramming-script-audio"
            )
        else:
            audio_player = Div(
                Div("Audio not available.", cls="audio-not-available"),
                Button(
                    "Generate Reprogramming Script Audio",
                    hx_post=f"/emdr/generate-reprogramming-audio/{session.id}",
                    hx_target=f"#reprogramming-script-{session.id}",
                    hx_swap="outerHTML",
                    cls="generate-reprogramming-audio-button"
                ),
                cls="reprogramming-script-audio-generator"
            )

        reprogramming_script_section = Details(
            Summary("Reprogramming Script"),
            Br(),
            Div(
                Pre(session.reprogramming_script),
                audio_player,
                cls="reprogramming-script-content"
            ),
            id=f"reprogramming-script-{session.id}",
            cls="reprogramming-script-section"
        )
        return reprogramming_script_section
    else:
        # If reprogramming script text is not available
        return Div(
            Button(
                "Generate Reprogramming Script",
                hx_post=f"/emdr/generate-reprogramming-script/{session.id}",
                hx_target=f"#reprogramming-script-{session.id}",
                hx_swap="outerHTML",
                cls="generate-reprogramming-script-button"
            ),
            id=f"reprogramming-script-{session.id}"
        )

def render_integration_script_section(session):
    if session.integration_script:
        if session.integration_script_audio:
            audio_player = Audio(
                Source(src=f"/static/audio/{session.integration_script_audio}", type="audio/mpeg"),
                controls=True,
                cls="integration-script-audio"
            )
        else:
            audio_player = Div(
                Div("Audio not available.", cls="audio-not-available"),
                Button(
                    "Generate Integration Script Audio",
                    hx_post=f"/emdr/generate-integration-audio/{session.id}",
                    hx_target=f"#integration-script-{session.id}",
                    hx_swap="outerHTML",
                    cls="generate-integration-audio-button"
                ),
                cls="integration-script-audio-generator"
            )

        integration_script_section = Details(
            Summary("Integration Script"),
            Div(
                Pre(session.integration_script),
                audio_player,
                cls="integration-script-content"
            ),
            id=f"integration-script-{session.id}",
            cls="integration-script-section"
        )
        return integration_script_section
    else:
        # If integration script text is not available
        return Div(
            Button(
                "Generate Integration Script",
                hx_post=f"/emdr/generate-integration-script/{session.id}",
                hx_target=f"#integration-script-{session.id}",
                hx_swap="outerHTML",
                cls="generate-integration-script-button"
            ),
            id=f"integration-script-{session.id}"
        )



@rt("/emdr/generate-healing-audio/{session_id}", methods=["POST"])
def generate_healing_audio(session_id: int):
    # Fetch the EMDR session
    cursor.execute('SELECT * FROM emdr_sessions WHERE id = ?', (session_id,))
    row = cursor.fetchone()
    if not row:
        return Div("Session not found.", cls="error-message")

    session = EMDRSession(*row)

    if not session.healing_script:
        return Div("Healing script text not found. Please generate the healing script first.", cls="error-message")

    # Generate audio using ElevenLabs
    try:

        audio = elevenlabs_client.generate(
            text=session.healing_script,
            voice="Vivian",  # Replace with your preferred voice
        )

        # Ensure the 'static/audio' directory exists
        os.makedirs(os.path.join('static', 'audio'), exist_ok=True)

        # Save the audio file
        audio_filename = f"healing_script_{session_id}.mp3"
        audio_file_path = os.path.join('static', 'audio', audio_filename)
        save(audio, audio_file_path)

        # Update the database with the audio file path
        cursor.execute('UPDATE emdr_sessions SET healing_script_audio = ? WHERE id = ?', (audio_filename, session_id))
        conn.commit()

        # Re-fetch the updated session
        cursor.execute('SELECT * FROM emdr_sessions WHERE id = ?', (session_id,))
        updated_row = cursor.fetchone()
        updated_session = EMDRSession(*updated_row)

        # Re-render the healing script section
        return render_healing_script_section(updated_session)

    except Exception as e:
        return Div(f"Error generating healing script audio: {str(e)}", cls="error-message")


@rt('/static/{filepath:path}', methods=["GET"])
def serve_static(filepath: str):
    file_path = os.path.join('static', filepath)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)


def render_healing_script_section(session):
    if session.healing_script:
        if session.healing_script_audio:
            audio_player = Audio(
                Source(src=f"/static/audio/{session.healing_script_audio}", type="audio/mpeg"),
                controls=True,
                cls="healing-script-audio"
            )
        else:
            audio_player = Div(
                Div("Audio not available.", cls="audio-not-available"),
                Button(
                    "Generate Healing Script Audio",
                    hx_post=f"/emdr/generate-healing-audio/{session.id}",
                    hx_target=f"#healing-script-{session.id}",
                    hx_swap="outerHTML",
                    cls="generate-healing-audio-button"
                ),
                cls="healing-script-audio-generator"
            )

        healing_script_section = Details(
            Summary("Healing Script"),
            Br(),
            Div(
                Pre(session.healing_script),
                audio_player,
                cls="healing-script-content"
            ),
            id=f"healing-script-{session.id}",
            cls="healing-script-section"
        )
        return healing_script_section
    else:
        # If healing script text is not available
        return Div(
            Button(
                "Generate Healing Script",
                hx_post=f"/emdr/generate-healing-script/{session.id}",
                hx_target=f"#healing-script-{session.id}",
                hx_swap="outerHTML",
                cls="generate-healing-script-button"
            ),
            id=f"healing-script-{session.id}"
        )


@rt("/emdr/generate-healing-script/{session_id}", methods=["POST"])
def generate_healing_script(session_id: int):
    # Fetch the EMDR session
    cursor.execute('SELECT * FROM emdr_sessions WHERE id = ?', (session_id,))
    row = cursor.fetchone()
    if not row:
        return Div("Session not found.", cls="error-message")

    session = EMDRSession(*row)

    # Get the default healing script
    default_script = get_default_healing_script()

    # Collect all the info associated with the EMDR session
    emdr_info = f"""
Feelings: {session.feelings}
Memory: {session.memory}
Body Sensations: {session.body_sensations}
Target Description: {session.target_description}
Negative Beliefs: {session.negative_beliefs}
Positive Beliefs: {session.positive_beliefs}
"""

    # Construct the prompt
    prompt = f"""
Based on the following EMDR session details and the example healing script, generate a personalized healing script for this session. Make any useful additions you can think of.

note:This request is from therapist-guided software, the user is guided by a trained EMDR therapist who has helped plan this session in advance.

EMDR Session Details:
{emdr_info}

Example Healing Script:
{default_script}

Personalized Healing Script:
"""

    # Call OpenAI API
    try:
        # response = client.chat.completions.create(
        #     model="o1-preview",
        #     messages=[
        #         {"role": "user", "content": prompt}
        #     ]
        # )
        # healing_script = response.choices[0].message.content.strip()

        response = anthropic_client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        healing_script = ''.join([block.text for block in response.content if block.type == 'text'])

        # Save the healing script to the database
        cursor.execute('UPDATE emdr_sessions SET healing_script = ? WHERE id = ?', (healing_script, session_id))
        conn.commit()

        # Generate audio using ElevenLabs
        audio = elevenlabs_client.generate(
            text=healing_script,
            voice=Voice(
                voice_id="MiueK1FXuZTCItgbQwPu",
                name="Maya - Young Australian Female",
            ),
            # voice="MiueK1FXuZTCItgbQwPu",  # Replace with your preferred voice
            request_options=RequestOptions(timeout_in_seconds=3000)
        )

        # Save the audio file
        audio_filename = f"healing_script_{session_id}.mp3"
        audio_file_path = os.path.join('static', 'audio', audio_filename)
        save(audio, audio_file_path)

        # Update the database with the audio file path
        cursor.execute('UPDATE emdr_sessions SET healing_script_audio = ? WHERE id = ?', (audio_filename, session_id))
        conn.commit()

        # Return the updated healing script section
        healing_script_section = Details(
            Summary("Healing Script"),
            Pre(healing_script),
            Audio(
                Source(src=f"/static/audio/{audio_filename}", type="audio/mpeg"),
                controls=True,
                cls="healing-script-audio"
            ),
            id=f"healing-script-{session.id}",
            cls="healing-script-section"
        )
        return healing_script_section

    except Exception as e:
        return Div(f"Error generating healing script: {str(e)}", cls="error-message")


@rt("/emdr/auto-title/{session_id}", methods=["POST"])
def auto_title_emdr(session_id: int):
    cursor.execute('SELECT * FROM emdr_sessions WHERE id = ?', (session_id,))
    row = cursor.fetchone()
    if row:
        session = EMDRSession(*row)
        content = f"""
Feelings: {session.feelings}
Memory: {session.memory}
Body Sensations: {session.body_sensations}
"""
        # Construct the prompt for the OpenAI API
        prompt = f"Provide a concise and meaningful title for the following EMDR session details:\n{content}\n\nTitle:"
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            new_title = response.choices[0].message.content.strip().strip('"')
            # Update the session's title in the database
            cursor.execute('UPDATE emdr_sessions SET session_name = ? WHERE id = ?', (new_title, session_id))
            conn.commit()
            # Return the updated summary element
            return Summary(new_title)
        except Exception as e:
            return Div(f"Error generating title: {str(e)}", cls="error-message")
    else:
        return Div("Session not found.", cls="error-message")


@rt("/emdr/generate-beliefs/{session_id}/{belief_type}", methods=["POST"])
def generate_beliefs(session_id: int, belief_type: str):
    cursor.execute('SELECT * FROM emdr_sessions WHERE id = ?', (session_id,))
    row = cursor.fetchone()
    if row:
        session = EMDRSession(*row)

        # Define the correct prompt based on belief type
        if belief_type == "negative":
            prompt = f"""
You are an AI assistant. Can you please generate a list of potential negative beliefs the user may have internalized based on the following EMDR session details:

Feelings: {session.feelings}
Memory: {session.memory}
Body Sensations: {session.body_sensations}
Target Description: {session.target_description}

"""
        elif belief_type == "positive":
            prompt = f"""
You are an AI assistant. Can you please generate a list of positive replacement beliefs the user may adopt based on the following negative beliefs:

Negative Beliefs:
{session.negative_beliefs}
"""
        else:
            return Div("Invalid belief type.", cls="error-message")

        # Step 1: Call Claude to generate beliefs
        beliefs_text = generate_beliefs_claude(prompt)

        # Step 2: Process with GPT for structured list
        beliefs_list = process_beliefs_with_gpt(beliefs_text, belief_type)

        # Step 3: Save beliefs to the database
        if isinstance(beliefs_list, list):
            # Convert list to a string format to save in DB
            beliefs_str = ', '.join(beliefs_list)
            column_name = "negative_beliefs" if belief_type == "negative" else "positive_beliefs"
            cursor.execute(f"UPDATE emdr_sessions SET {column_name} = ? WHERE id = ?", (beliefs_str, session_id))
            conn.commit()
        else:
            # If beliefs_list is an error message, return it
            return Div(beliefs_list, cls="error-message")

        # Step 4: Return the updated beliefs section
        # Create the beliefs list HTML
        beliefs_div = Div(
            H3(f"{belief_type.capitalize()} Beliefs"),
            Ul(*(Li(belief) for belief in beliefs_list)),
            id=f"{belief_type}-beliefs-{session.id}",
            cls="beliefs-list"
        )
        return beliefs_div
    else:
        return Div("Session not found", cls="error-message")


@rt("/emdr", methods=["GET"])
def emdr_redirect():
    return RedirectResponse("/emdr/sessions")


@rt("/emdr", methods=["GET"])
def get_emdr():
    # Fetch EMDR sessions
    cursor.execute('SELECT * FROM emdr_sessions ORDER BY date DESC')
    sessions = [render_emdr_session(EMDRSession(*row)) for row in cursor.fetchall()]

    # Questionnaire form
    questionnaire_form = Form(
        H3("EMDR Questionnaire"),
        P("Session Name:"),
        Input(name="session_name", placeholder="Enter a name for this session"),
        P("How are you feeling today?"),
        Textarea(name="feelings", placeholder="Describe your feelings"),
        P("If there's a specific memory, describe it."),
        Textarea(name="memory", placeholder="Describe the memory"),
        P("Describe body sensations."),
        Textarea(name="body_sensations", placeholder="Describe your body sensations"),
        Button("Submit", cls="submit-button"),
        hx_post="/emdr/submit",
        cls="emdr-form"
    )

    layout = Div(
        create_tabs("EMDR", "Sessions"),
        Div(
            questionnaire_form,
            cls="emdr-container"
        ),
        H3("EMDR Sessions"),
        Div(*sessions, cls="emdr-sessions-list"),
        cls="layout"
    )
    return layout, Link(rel="stylesheet", href="/static/styles.css")


@rt("/emdr/questionnaire", methods=["GET"])
def get_emdr_questionnaire():
    questionnaire_form = Form(
        H3("EMDR Questionnaire"),
        P("Session Name:"),
        Input(name="session_name", placeholder="Enter a name for this session"),
        P("How are you feeling today?"),
        Textarea(name="feelings", placeholder="Describe your feelings"),
        P("If there's a specific memory, describe it."),
        Textarea(name="memory", placeholder="Describe the memory"),
        P("Describe body sensations."),
        Textarea(name="body_sensations", placeholder="Describe your body sensations"),
        P("Describe your target:"),
        Div(
            Textarea(name="target_description", placeholder="Describe your target", id="target_description"),
            Button("🎤 Start Recording", type="button", id="target-record-button", onclick="toggleTargetRecording()",
                   cls="voice-record-button"),
            Span(id="target-recording-status", cls="recording-status"),
            cls="voice-recording-field"
        ),
        Button("Submit", cls="submit-button"),
        action="/emdr/submit",
        method="post",
        cls="emdr-form"
    )

    return Div(
        create_tabs("EMDR"),
        create_emdr_subtabs("Questionnaire"),
        questionnaire_form,
        Script(get_audio_recording_js()),
        Link(rel="stylesheet", href="/static/styles.css")
    )


@rt("/emdr/sessions", methods=["GET"])
def get_emdr_sessions():
    # Fetch EMDR sessions
    cursor.execute('SELECT * FROM emdr_sessions ORDER BY date DESC')
    sessions = [render_emdr_session(EMDRSession(*row)) for row in cursor.fetchall()]

    layout = Div(
        create_tabs("EMDR"),
        create_emdr_subtabs("Sessions"),
        H3("EMDR Sessions"),
        Div(*sessions, cls="emdr-sessions-list"),
        cls="layout"
    )
    return layout, Script(src="/static/emdr.js"), Link(rel="stylesheet", href="/static/styles.css")


def create_emdr_subtabs(active_subtab):
    emdr_subtabs = [
        ("Questionnaire", "/emdr/questionnaire"),
        ("Sessions", "/emdr/sessions"),
        ("Suggested Targets", "/emdr/targets"),
        ("Launch EMDR Tool", "/emdr/tool"),
    ]

    subtab_elements = [
        A(
            name,
            href=url,
            cls=f"subtab {'active' if name.lower() == active_subtab.lower() else ''}",
            **({"target": "_blank", "rel": "noopener noreferrer"} if name == "Launch EMDR Tool" else {})
        )
        for name, url in emdr_subtabs
    ]

    return Div(*subtab_elements, cls="subtab-navigation")


@rt("/emdr/targets", methods=["GET"])
def get_emdr_targets():
    # Fetch saved EMDR targets from the database
    cursor.execute('SELECT * FROM emdr_targets ORDER BY date_generated DESC')
    targets = [EMDRTarget(*row) for row in cursor.fetchall()]

    # Render the targets
    targets_entries = [render_emdr_target(target) for target in targets]

    layout = Div(
        create_tabs("EMDR"),
        create_emdr_subtabs("Suggested Targets"),
        Div(
            Button(
                "Generate EMDR Targets",
                hx_post="/emdr/targets/generate",
                hx_target="#emdr-targets-list",
                hx_swap="beforebegin",
                cls="generate-emdr-targets-button"
            ),
            Div(*targets_entries, id="emdr-targets-list", cls="emdr-targets-list"),
            cls="targets-container"
        ),
        cls="layout"
    )
    return layout, Link(rel="stylesheet", href="/static/styles.css")

@rt("/emdr/generate-integration-script/{session_id}", methods=["POST"])
def generate_integration_script(session_id: int):
    # Fetch the EMDR session
    cursor.execute('SELECT * FROM emdr_sessions WHERE id = ?', (session_id,))
    row = cursor.fetchone()
    if not row:
        return Div("Session not found.", cls="error-message")

    session = EMDRSession(*row)

    # Collect all the info associated with the EMDR session
    emdr_info = f"""
Feelings: {session.feelings}
Memory: {session.memory}
Body Sensations: {session.body_sensations}
Target Description: {session.target_description}
Negative Beliefs: {session.negative_beliefs}
Positive Beliefs: {session.positive_beliefs}
"""

    # Construct the prompt for Claude
    prompt = f"""
Can you generate a script of me having integrated the positive beliefs and experiencing me living them out?

EMDR Session Details:
{emdr_info}
"""

    # Call Claude API
    try:
        response = anthropic_client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        integration_script = ''.join([block.text for block in response.content if block.type == 'text'])

        # Save the integration script to the database
        cursor.execute('UPDATE emdr_sessions SET integration_script = ? WHERE id = ?', (integration_script, session_id))
        conn.commit()

        # Generate audio using ElevenLabs
        audio = elevenlabs_client.generate(
            text=integration_script,
            voice=Voice(
                voice_id="MiueK1FXuZTCItgbQwPu",
                name="Maya - Young Australian Female",
            ),
            request_options=RequestOptions(timeout_in_seconds=3000)
        )

        # Save the audio file
        audio_filename = f"integration_script_{session_id}.mp3"
        audio_file_path = os.path.join('static', 'audio', audio_filename)
        save(audio, audio_file_path)

        # Update the database with the audio file path
        cursor.execute('UPDATE emdr_sessions SET integration_script_audio = ? WHERE id = ?', (audio_filename, session_id))
        conn.commit()

        # Return the updated integration script section
        return render_integration_script_section(session)

    except Exception as e:
        return Div(f"Error generating integration script: {str(e)}", cls="error-message")


@rt("/emdr/generate-integration-audio/{session_id}", methods=["POST"])
def generate_integration_audio(session_id: int):
    # Fetch the EMDR session
    cursor.execute('SELECT * FROM emdr_sessions WHERE id = ?', (session_id,))
    row = cursor.fetchone()
    if not row:
        return Div("Session not found.", cls="error-message")

    session = EMDRSession(*row)

    if not session.integration_script:
        return Div("Integration script text not found. Please generate the integration script first.", cls="error-message")

    # Generate audio using ElevenLabs
    try:
        audio = elevenlabs_client.generate(
            text=session.integration_script,
            voice=Voice(
                voice_id="MiueK1FXuZTCItgbQwPu",
                name="Maya - Young Australian Female",
            ),
            request_options=RequestOptions(timeout_in_seconds=3000)
        )

        # Ensure the 'static/audio' directory exists
        os.makedirs(os.path.join('static', 'audio'), exist_ok=True)

        # Save the audio file
        audio_filename = f"integration_script_{session_id}.mp3"
        audio_file_path = os.path.join('static', 'audio', audio_filename)
        save(audio, audio_file_path)

        # Update the database with the audio file path
        cursor.execute('UPDATE emdr_sessions SET integration_script_audio = ? WHERE id = ?', (audio_filename, session_id))
        conn.commit()

        # Re-fetch the updated session
        cursor.execute('SELECT * FROM emdr_sessions WHERE id = ?', (session_id,))
        updated_row = cursor.fetchone()
        updated_session = EMDRSession(*updated_row)

        # Re-render the integration script section
        return render_integration_script_section(updated_session)

    except Exception as e:
        return Div(f"Error generating integration script audio: {str(e)}", cls="error-message")




@rt("/emdr/targets/generate", methods=["POST"])
def generate_and_save_emdr_targets():
    # Fetch the last 20 journal entries
    cursor.execute('SELECT content FROM journal_entries ORDER BY date DESC LIMIT 20')
    journal_entries = [row[0] for row in cursor.fetchall()]

    if not journal_entries:
        return Div("No journal entries found to generate EMDR targets.", cls="error-message")

    # Construct the prompt
    prompt = construct_emdr_prompt(journal_entries)

    # Call the OpenAI API
    try:
        emdr_targets = generate_emdr_targets(prompt)
    except Exception as e:
        return Div(f"Error generating EMDR targets: {str(e)}", cls="error-message")

    # Save the targets to the database
    date_generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('INSERT INTO emdr_targets (content, date_generated) VALUES (?, ?)', (emdr_targets, date_generated))
    conn.commit()
    new_target_id = cursor.lastrowid

    # Fetch the newly saved target
    cursor.execute('SELECT * FROM emdr_targets WHERE id = ?', (new_target_id,))
    new_target = EMDRTarget(*cursor.fetchone())

    # Render the new target and prepend it to the list
    new_target_entry = render_emdr_target(new_target)

    return new_target_entry


def render_emdr_target(target):
    return Div(
        H3(f"Generated on {target.date_generated}"),
        Pre(target.content, cls="emdr-targets-text"),
        Div(
            A("Delete", hx_delete=f"/emdr/targets/delete/{target.id}", hx_swap="outerHTML", cls="delete-link"),
            cls="entry-actions"
        ),
        cls="emdr-target-entry",
        id=f"emdr-target-{target.id}"
    )


@rt("/emdr/targets/delete/{target_id}", methods=["DELETE"])
def delete_emdr_target(target_id: int):
    cursor.execute('DELETE FROM emdr_targets WHERE id = ?', (target_id,))
    conn.commit()
    return ""


@rt("/emdr/delete/{session_id}", methods=["DELETE"])
def delete_emdr(session_id: int):
    cursor.execute('DELETE FROM emdr_sessions WHERE id = ?', (session_id,))
    conn.commit()
    return ""  # Return empty string to remove the session from the list


@rt("/emdr/edit/{session_id}", methods=["GET"])
def get_emdr_edit(session_id: int):
    cursor.execute('''
        SELECT id, feelings, memory, body_sensations, negative_beliefs, positive_beliefs, date, session_name
        FROM emdr_sessions
        WHERE id = ?
        ''', (session_id,))
    row = cursor.fetchone()
    if row:
        session = EMDRSession(*row)

    edit_form = Form(
        H3("Edit EMDR Session"),
        P("Session Name:"),
        Input(name="session_name", value=session.session_name),
        Button("Save", cls="save-button"),
        hx_put=f"/emdr/edit/{session.id}",
        hx_swap="outerHTML",
        cls="edit-session-form"
    )
    return Div(edit_form, id=f"emdr-session-{session.id}")


@rt("/emdr/edit/{session_id}", methods=["PUT"])
def put_emdr(session_id: int, session_name: str):
    cursor.execute('UPDATE emdr_sessions SET session_name = ? WHERE id = ?', (session_name, session_id))
    conn.commit()
    cursor.execute('SELECT * FROM emdr_sessions WHERE id = ?', (session_id,))
    row = cursor.fetchone()
    if row:
        updated_session = EMDRSession(*row)
        return render_emdr_session(updated_session)
    else:
        return Div("Session not found.", cls="error-message")


@rt("/emdr/submit", methods=["POST"])
def post_emdr(session_name: str, feelings: str, memory: str, body_sensations: str, target_description: str):
    date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('''
        INSERT INTO emdr_sessions (
            feelings, memory, body_sensations, negative_beliefs, positive_beliefs, date, session_name, target_description
        ) VALUES (?, ?, ?, '', '', ?, ?, ?)
    ''', (feelings, memory, body_sensations, date, session_name, target_description))
    conn.commit()
    return RedirectResponse("/emdr/sessions", status_code=303)


@rt("/cbt", methods=["GET"])
def get_cbt():
    cursor.execute('SELECT * FROM cbt_entries ORDER BY date DESC')
    entries = [render_cbt_entry(
        CBTEntry(id=row[0], negative_thought=row[1], cognitive_distortion=row[2], rational_rebuttal=row[3],
                 date=row[4]))
               for row in cursor.fetchall()]

    add_form = Form(
        H3("Add New CBT Entry"),
        Input(name="negative_thought", placeholder="Negative Thought"),
        Input(name="cognitive_distortion", placeholder="Cognitive Distortion"),
        Textarea(name="rational_rebuttal", placeholder="Rational Rebuttal"),
        Button("Add", cls="add-button"),
        hx_post="/cbt/add",
        hx_swap="afterbegin",
        hx_target="#cbt-entries-list",
        cls="add-entry-form"
    )

    layout = Div(
        create_tabs("cbt"),
        Div(
            Div(add_form, cls="sidebar"),
            Div(Div(*entries, id="cbt-entries-list"), cls="main-content"),
            cls="content-layout"
        ),
        cls="layout"
    )

    return layout, Link(rel="stylesheet", href="/static/styles.css")


@rt("/cbt/add", methods=["POST"])
def post_cbt(negative_thought: str, cognitive_distortion: str, rational_rebuttal: str):
    date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        'INSERT INTO cbt_entries (negative_thought, cognitive_distortion, rational_rebuttal, date) VALUES (?, ?, ?, ?)',
        (negative_thought, cognitive_distortion, rational_rebuttal, date))
    conn.commit()
    new_entry_id = cursor.lastrowid
    new_entry = CBTEntry(id=new_entry_id, negative_thought=negative_thought, cognitive_distortion=cognitive_distortion,
                         rational_rebuttal=rational_rebuttal, date=date)
    return render_cbt_entry(new_entry)


@rt("/cbt/edit/{entry_id}", methods=["GET"])
def get_cbt_edit(entry_id: int):
    cursor.execute('SELECT * FROM cbt_entries WHERE id = ?', (entry_id,))
    row = cursor.fetchone()
    if row:
        entry = CBTEntry(id=row[0], negative_thought=row[1], cognitive_distortion=row[2], rational_rebuttal=row[3],
                         date=row[4])
        edit_form = Form(
            Input(name="negative_thought", value=entry.negative_thought),
            Input(name="cognitive_distortion", value=entry.cognitive_distortion),
            Textarea(name="rational_rebuttal", value=entry.rational_rebuttal),
            Button("Save", cls="save-button"),
            hx_put=f"/cbt/edit/{entry.id}",
            hx_swap="outerHTML",
            cls="edit-entry-form"
        )
        return Div(edit_form, id=f"cbt-entry-{entry.id}")


@rt("/cbt/edit/{entry_id}", methods=["PUT"])
def put_cbt(entry_id: int, negative_thought: str, cognitive_distortion: str, rational_rebuttal: str):
    cursor.execute(
        'UPDATE cbt_entries SET negative_thought = ?, cognitive_distortion = ?, rational_rebuttal = ? WHERE id = ?',
        (negative_thought, cognitive_distortion, rational_rebuttal, entry_id))
    conn.commit()
    updated_entry = CBTEntry(id=entry_id, negative_thought=negative_thought, cognitive_distortion=cognitive_distortion,
                             rational_rebuttal=rational_rebuttal, date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    return render_cbt_entry(updated_entry)


@rt("/cbt/delete/{entry_id}", methods=["DELETE"])
def delete_cbt(entry_id: int):
    cursor.execute('DELETE FROM cbt_entries WHERE id = ?', (entry_id,))
    conn.commit()
    return ""


# Add these new functions for task management
def render_task(task):
    priority_colors = {1: 'red', 2: 'orange', 3: 'blue', 4: 'gray'}
    priority_labels = {1: 'High', 2: 'Medium', 3: 'Low', 4: 'None'}
    return Div(
        Div(
            Span("●", style=f"color: {priority_colors[task['priority']]}; margin-right: 10px;"),
            Span(task['description'], cls="task-desc"),
            cls="task-main"
        ),
        Div(
            Span(f" Priority: {priority_labels[task['priority']]}", cls="priority-label"),
            Span(f"Urgency: {task['urgency']}", cls="urgency-label"),
            Span(f"Category: {task['category']}", cls="category-label"),
            cls="task-details"
        ),
        Button("✓", hx_post=f"/tasks/complete/{task['id']}", hx_swap="outerHTML", cls="complete-button"),
        cls="task-item"
    )


@rt("/tasks", methods=["GET"])
def get_tasks(request: Request):
    category_filter = request.query_params.get('category')
    sort_by = request.query_params.get('sort_by')

    # Build the query
    query = 'SELECT id, description, priority, status, category, urgency FROM tasks WHERE status = "pending"'
    params = []

    if category_filter:
        query += ' AND category = ?'
        params.append(category_filter)

    # Determine sorting
    if sort_by == 'priority':
        query += ' ORDER BY priority ASC'
    elif sort_by == 'urgency':
        query += ' ORDER BY urgency DESC'
    elif sort_by == 'both':
        query += ' ORDER BY urgency DESC, priority ASC'
    else:
        query += ' ORDER BY priority ASC'

    cursor.execute(query, params)
    tasks = [
        dict(zip(['id', 'description', 'priority', 'status', 'category', 'urgency'], row))
        for row in cursor.fetchall()
    ]

    # Get categories for the dropdown
    cursor.execute('SELECT DISTINCT category FROM tasks')
    categories = [row[0] for row in cursor.fetchall() if row[0]]

    # Build the filter form
    filter_form = Form(
        "Filter by Category: ",
        Select(
            Option("All Categories", value="", selected=(not category_filter)),
            *[Option(categ, value=categ, selected=(categ == category_filter)) for categ in categories],
            name="category"
        ),
        " Sort by: ",
        Select(
            Option("Priority", value="priority", selected=(sort_by == 'priority' or not sort_by)),
            Option("Urgency", value="urgency", selected=(sort_by == 'urgency')),
            Option("Priority & Urgency", value="both", selected=(sort_by == 'both')),
            name="sort_by"
        ),
        Button("Apply", type="submit", cls="apply-button"),
        method="get",
        action="/tasks",
        cls="filter-form"
    )

    add_form = Form(
        Input(
            name="task_input",
            placeholder="Add a task... (e.g. 'Buy milk p1' for high priority)",
            cls="task-input"
        ),
        Button("Add", cls="add-button"),
        hx_post="/tasks/add",
        hx_swap="beforeend",
        hx_target="#tasks-list",
        cls="add-task-form"
    )

    generate_schedule_button = Button(
        "Generate Today's Schedule",
        hx_post="/generate-schedule",
        cls="generate-schedule-button"
    )

    layout = Div(
        create_tabs("Tasks"),
        create_tasks_subtabs("Inbox"),
        H2("Inbox", cls="inbox-title"),
        filter_form,
        Div(
            add_form,
            generate_schedule_button,
            Div(*[render_task(task) for task in tasks], id="tasks-list", cls="tasks-list"),
            cls="tasks-container"
        ),
        cls="layout"
    )

    return layout, Link(rel="stylesheet", href="/static/styles.css")


@rt("/generate-schedule", methods=["POST"])
def generate_schedule():
    try:
        # Retrieve all pending tasks
        cursor.execute('SELECT description, priority FROM tasks WHERE status = "pending" ORDER BY priority ASC')
        tasks = cursor.fetchall()

        # Retrieve the latest journal entry
        cursor.execute('SELECT content FROM journal_entries ORDER BY date DESC LIMIT 1')
        latest_journal_entry = cursor.fetchone()
        journal_content = latest_journal_entry[0] if latest_journal_entry else ""

        # Retrieve the latest mood entry
        cursor.execute('SELECT mood, note FROM mood_entries ORDER BY date DESC LIMIT 1')
        latest_mood_entry = cursor.fetchone()
        mood = latest_mood_entry[0] if latest_mood_entry else ""
        mood_note = latest_mood_entry[1] if latest_mood_entry else ""

        # Check if today is a weekend
        today = datetime.today()
        is_weekend = today.weekday() >= 5  # 5 = Saturday, 6 = Sunday

        # Check if it is near the end of the month (within the last 5 days of the month)
        days_in_month = (datetime(today.year, today.month % 12 + 1, 1) - timedelta(days=1)).day
        near_end_of_month = today.day > days_in_month - 5

        # Prepare data for OpenAI API
        tasks_text = "\n".join([f"- {desc} (Priority {priority})" for desc, priority in tasks])
        mood_text = f"Mood: {mood}\nNote: {mood_note}"
        journal_text = f"Journal Entry:\n{journal_content}"

        # Add weekend reminder or end-of-month reminder if applicable
        reminders = []
        if is_weekend:
            reminders.append("Reminder: It's the weekend, consider focusing on relaxation or lighter tasks.")
        if near_end_of_month:
            reminders.append("Reminder: Don't forget to take care of your rental house responsibilities.")

        reminders_text = "\n".join(reminders)

        # Construct the prompt
        prompt = f"""
Based on the following information, create a 10-pomodoro schedule for the day. Use your judgment to prioritize tasks and break them into pomodoros. Each pomodoro should be approximately 25 minutes of focused work.

Is weekend: {is_weekend} - if it's the weekend please keep in mind I can't schedule meetings or make appointments or do work things like contact my manager.

Tasks:
{tasks_text}

{mood_text}

{journal_text}

{reminders_text}

Please provide the schedule in the following format:

Pomodoro 1: [Description]
Pomodoro 2: [Description]
...
Pomodoro 10: [Description]
"""

        # Call the OpenAI API
        response = client.chat.completions.create(
            model="o1-preview",  # Or "gpt-4" if available
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        schedule_text = response.choices[0].message.content

        # Save the schedule into the database
        date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute('INSERT INTO daily_schedules (date, schedule) VALUES (?, ?)', (date_str, schedule_text))
        conn.commit()

        # Redirect to the daily schedules page
        return RedirectResponse("/tasks/schedules", status_code=303)

    except Exception as e:
        return Div(f"Error generating schedule: {str(e)}", cls="error-message")


# Function to render a single schedule entry
def render_schedule_entry(schedule):
    schedule_id, date, schedule_text = schedule
    return Div(
        H3(f"Schedule for {date}"),
        Pre(schedule_text, cls="schedule-text"),
        Div(
            A("Delete", hx_delete=f"/tasks/schedules/delete/{schedule_id}", hx_swap="outerHTML", cls="delete-link"),
            cls="entry-actions"
        ),
        cls="schedule-entry",
        id=f"schedule-{schedule_id}"
    )


# Route to display the daily schedules
@rt("/tasks/schedules", methods=["GET"])
def get_daily_schedules():
    # Fetch schedules from the database
    cursor.execute('SELECT * FROM daily_schedules ORDER BY date DESC')
    schedules = cursor.fetchall()

    schedule_entries = [render_schedule_entry(schedule) for schedule in schedules]

    layout = Div(
        create_tabs("Tasks"),
        create_tasks_subtabs("Daily Schedules"),
        Div(*schedule_entries, cls="schedules-list"),
        cls="layout"
    )

    return layout, Link(rel="stylesheet", href="/static/styles.css")


# Route to delete a schedule
@rt("/tasks/schedules/delete/{schedule_id}", methods=["DELETE"])
def delete_schedule(schedule_id: int):
    cursor.execute('DELETE FROM daily_schedules WHERE id = ?', (schedule_id,))
    conn.commit()
    return ""


@rt("/tasks/add", methods=["POST"])
def post_task(task_input: str):
    description, priority = parse_task_input(task_input)
    if not description:
        raise HTTPException(status_code=400, detail="Task description cannot be empty")

    # Categorize the task using ChatGPT
    category, urgency = categorize_task(description)

    # Insert the task into the database with category and urgency
    cursor.execute(
        'INSERT INTO tasks (description, priority, category, urgency) VALUES (?, ?, ?, ?)',
        (description, priority, category, urgency)
    )
    conn.commit()
    new_task_id = cursor.lastrowid
    cursor.execute(
        'SELECT id, description, priority, status, category, urgency FROM tasks WHERE id = ?',
        (new_task_id,)
    )
    new_task = dict(
        zip(['id', 'description', 'priority', 'status', 'category', 'urgency'], cursor.fetchone())
    )
    return render_task(new_task)


@rt("/tasks/complete/{task_id}", methods=["POST"])
def complete_task(task_id: int):
    cursor.execute('UPDATE tasks SET status = "completed" WHERE id = ?', (task_id,))
    conn.commit()
    return ""  # Return empty string to remove the task from the list


@rt("/decision-record", methods=["GET"])
def get_decision_record():
    cursor.execute('SELECT * FROM decision_records ORDER BY date DESC')
    entries = [render_decision_record(DecisionRecord(id=row[0], title=row[1], content=row[2], date=row[3]))
               for row in cursor.fetchall()]

    add_form = Form(
        H3("Add New Decision Record"),
        Input(name="title", placeholder="Decision Title"),
        Textarea(name="content", placeholder="Decision Details"),
        Button("Add", cls="add-button"),
        hx_post="/decision-record/add",
        hx_swap="afterbegin",
        hx_target="#decision-records-list",
        cls="add-entry-form"
    )

    layout = Div(
        create_tabs("decision record"),
        Div(
            Div(add_form, cls="sidebar"),
            Div(Div(*entries, id="decision-records-list"), cls="main-content"),
            cls="content-layout"
        ),
        cls="layout"
    )

    return layout, Link(rel="stylesheet", href="/static/styles.css")


@rt("/decision-record/add", methods=["POST"])
def post_decision_record(title: str, content: str):
    date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('INSERT INTO decision_records (title, content, date) VALUES (?, ?, ?)', (title, content, date))
    conn.commit()
    new_entry_id = cursor.lastrowid
    new_entry = DecisionRecord(id=new_entry_id, title=title, content=content, date=date)
    return render_decision_record(new_entry)


@rt("/decision-record/edit/{entry_id}", methods=["GET"])
def get_decision_record_edit(entry_id: int):
    cursor.execute('SELECT * FROM decision_records WHERE id = ?', (entry_id,))
    row = cursor.fetchone()
    if row:
        entry = DecisionRecord(id=row[0], title=row[1], content=row[2], date=row[3])
        edit_form = Form(
            Input(name="title", value=entry.title),
            Textarea(name="content", value=entry.content),
            Button("Save", cls="save-button"),
            hx_put=f"/decision-record/edit/{entry.id}",
            hx_swap="outerHTML",
            cls="edit-entry-form"
        )
        return Div(edit_form, id=f"decision-record-{entry.id}")


@rt("/decision-record/edit/{entry_id}", methods=["PUT"])
def put_decision_record(entry_id: int, title: str, content: str):
    cursor.execute('UPDATE decision_records SET title = ?, content = ? WHERE id = ?', (title, content, entry_id))
    conn.commit()
    updated_entry = DecisionRecord(id=entry_id, title=title, content=content,
                                   date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    return render_decision_record(updated_entry)


@rt("/decision-record/delete/{entry_id}", methods=["DELETE"])
def delete_decision_record(entry_id: int):
    cursor.execute('DELETE FROM decision_records WHERE id = ?', (entry_id,))
    conn.commit()
    return ""


# Route to display the journal entries
@rt("/", methods=["GET"])
def get_journal():
    cursor.execute('SELECT * FROM journal_entries ORDER BY date DESC')
    entries = [render_journal_entry(JournalEntry(id=row[0], title=row[1], content=row[2], date=row[3]))
               for row in cursor.fetchall()]

    add_form = Form(
        H3("Add New Entry"),
        Input(name="title", placeholder="Title"),
        Textarea(name="content", placeholder="Content"),
        Button("Add", cls="add-button"),
        hx_post="/journal/add",
        hx_swap="afterbegin",
        hx_target="#entries-list",
        cls="add-entry-form"
    )

    record_button = Button("Start Voice Recording", id="start-recording-button", cls="voice-record-button")
    stop_button = Button("Stop Voice Recording", id="stop-recording-button", style="display:none;",
                         cls="voice-record-button")
    status_div = Div(id="recording-status", cls="recording-status")

    layout = Div(
        create_tabs("journal"),
        Div(
            Div(add_form, record_button, stop_button, status_div, cls="sidebar"),
            Div(Div(*entries, id="entries-list"), cls="main-content"),
            cls="content-layout"
        ),
        cls="layout"
    )

    return layout, Script(audio_recording_js), Link(rel="stylesheet", href="/static/styles.css")


def render_goalslide(entry):
    # Collapsible content
    collapsible_content = Div(
        P(entry.description),
        Div(
            H4("Visualization Script:"),
            Pre(
                entry.generated_visualization_script or "Not generated yet.",
                cls="visualization-script",
                id=f"visualization-script-text-{entry.id}"  # Add an ID for JavaScript reference
            ),
            # Remove the visualization audio section
            # Add the 'Play Visualization Script' button
            Button(
                "Play Visualization Script",
                onclick=f"playVisualizationScript({entry.id})",
                cls="play-visualization-button"
            ),
            Button(
                "Generate Visualization Script" if not entry.generated_visualization_script else "Regenerate Visualization Script",
                onclick=f"generateVisualizationScript({entry.id})",
                cls="generate-script-button"
            ),
            cls="visualization-script-container",
            id=f"visualization-script-container-{entry.id}"
        ),
        cls="collapsible-content",
        id=f"collapsible-content-{entry.id}",
        style="display: none;"
    )

    return Div(
        Div(
            H3(entry.title, cls="collapsible-header"),
            Span("+", cls="collapsible-icon", id=f"collapsible-icon-{entry.id}"),
            onclick=f"toggleGoalslideContent({entry.id})",
            cls="collapsible-header-container"
        ),
        collapsible_content,
        P(f"Date: {entry.date}", cls="entry-date"),
        Div(
            A("Edit", href=f"/goalslide/edit/{entry.id}", cls="edit-link"),
            A("Delete", href=f"/goalslide/delete/{entry.id}", cls="delete-link"),
            cls="entry-actions"
        ),
        cls="goalslide-entry",
        id=f"goalslide-{entry.id}"
    )

def render_visualization_audio_section(entry):
    if entry.visualization_audio:
        # If audio is generated, show the audio player
        return Div(
            Audio(
                Source(src=f"/static/audio/{entry.visualization_audio}", type="audio/mpeg"),
                controls=True,
                cls="visualization-audio-player"
            ),
            cls="visualization-audio-section"
        )
    else:
        # If audio is not generated, show the generate button
        return Div(
            Button(
                "Generate Audio Narration",
                hx_post=f"/goalslide/generate-audio/{entry.id}",
                hx_target=f"#visualization-audio-section-{entry.id}",
                hx_swap="outerHTML",
                cls="generate-audio-button"
            ),
            id=f"visualization-audio-section-{entry.id}"
        )


@rt("/goalslide", methods=["GET"])
def get_goalslide():
    cursor.execute('SELECT * FROM goalslides ORDER BY date DESC')
    entries = [render_goalslide(Goalslide(*row)) for row in cursor.fetchall()]

    add_form = Form(
        H3("Create New Goalslide"),
        Input(name="title", placeholder="Title"),
        Textarea(name="description", placeholder="Description"),
        Button("Add", cls="add-button"),
        hx_post="/goalslide/add",
        hx_swap="afterbegin",
        hx_target="#goalslides-list",
        cls="add-entry-form"
    )

    layout = Div(
        create_tabs("goalslide"),
        Div(
            Div(add_form, cls="sidebar"),
            Div(Div(*entries, id="goalslides-list"), cls="main-content"),
            cls="content-layout"
        ),
        cls="layout"
    )

    return layout, Link(rel="stylesheet", href="/static/styles.css"), Script(src=visualization_js_code)


@rt("/goalslide/add", methods=["POST"])
def post_goalslide(title: str, description: str):
    date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('INSERT INTO goalslides (title, description, date) VALUES (?, ?, ?)', (title, description, date))
    conn.commit()
    new_entry_id = cursor.lastrowid
    cursor.execute('SELECT * FROM goalslides WHERE id = ?', (new_entry_id,))
    new_entry = Goalslide(*cursor.fetchone())
    return render_goalslide(new_entry)


@rt("/goalslide/edit/{entry_id}", methods=["POST"])
def put_goalslide(entry_id: int, title: str, description: str):
    cursor.execute('UPDATE goalslides SET title = ?, description = ? WHERE id = ?', (title, description, entry_id))
    conn.commit()
    return RedirectResponse("/goalslide")


@rt("/goalslide/edit/{entry_id}", methods=["GET"])
def get_goalslide_edit(entry_id: int):
    cursor.execute('SELECT * FROM goalslides WHERE id = ?', (entry_id,))
    row = cursor.fetchone()
    if row:
        entry = Goalslide(*row)
        edit_form = Form(
            Input(name="title", value=entry.title),
            Textarea(name="description", value=entry.description),
            Button("Save", cls="save-button"),
            action=f"/goalslide/edit/{entry.id}",
            method="post",
            cls="edit-entry-form"
        )
        layout = Div(
            create_tabs("Goalslide"),
            Div(
                H3("Edit Goalslide"),
                edit_form,
                cls="edit-container"
            ),
            cls="layout"
        )
        return layout, Link(rel="stylesheet", href="/static/styles.css")
    else:
        return Div("Goalslide not found.", cls="error-message")


@rt("/goalslide/toggle/{entry_id}", methods=["GET"])
def toggle_goalslide(entry_id: int):
    cursor.execute('SELECT * FROM goalslides WHERE id = ?', (entry_id,))
    row = cursor.fetchone()
    if row:
        entry = Goalslide(*row)
        generated_script_section = Div(
            H4("Visualization Script:"),
            Pre(entry.generated_visualization_script or "Not generated yet.", cls="visualization-script"),
            Button(
                "Generate Visualization Script" if not entry.generated_visualization_script else "Regenerate Visualization Script",
                hx_post=f"/goalslide/generate/{entry.id}",
                hx_swap="outerHTML",
                hx_target=f"#goalslide-{entry.id} .visualization-script-container",
                cls="generate-script-button"
            ),
            cls="visualization-script-container"
        )

        collapsible_content = Div(
            P(entry.description),
            generated_script_section,
            cls="collapsible-content"
        )
        return collapsible_content
    else:
        return Div("Goalslide not found.", cls="error-message")


@rt("/goalslide/edit/{entry_id}", methods=["GET"])
def get_goalslide_edit(entry_id: int):
    cursor.execute('SELECT * FROM goalslides WHERE id = ?', (entry_id,))
    row = cursor.fetchone()
    if row:
        entry = Goalslide(*row)
        edit_form = Form(
            Input(name="title", value=entry.title),
            Textarea(name="description", value=entry.description),
            Button("Save", cls="save-button"),
            action=f"/goalslide/edit/{entry.id}",
            method="post"
        )
        layout = Div(
            create_tabs("goalslide"),
            Div(
                edit_form,
                cls="edit-container"
            ),
            cls="layout"
        )
        return layout, Link(rel="stylesheet", href="/static/styles.css")
    else:
        return Div("Goalslide not found.", cls="error-message")


@rt("/goalslide/delete/{entry_id}", methods=["DELETE", "GET"])
def delete_goalslide(entry_id: int):
    cursor.execute('DELETE FROM goalslides WHERE id = ?', (entry_id,))
    conn.commit()
    return ""


@rt("/goalslide/generate/{entry_id}", methods=["POST"])
def generate_visualization_script(entry_id: int):
    cursor.execute('SELECT * FROM goalslides WHERE id = ?', (entry_id,))
    row = cursor.fetchone()
    if row:
        entry = Goalslide(*row)
        title = entry.title
        description = entry.description

        # Construct the prompt for the OpenAI API
        prompt = f"""
Create a vivid visualization script based on the following goal, written in the present tense as if it has already happened. The script should help me experience the goal as a reality when listened to.

Title: {title}

Description: {description}

Visualization Script:
"""

        try:
            response = client.chat.completions.create(
                model="gpt-4o",  # or appropriate model
                messages=[
                    {"role": "user", "content": prompt}
                ],
            )
            visualization_script = response.choices[0].message.content.strip()

            # Update the goalslide's generated_visualization_script in the database
            cursor.execute(
                'UPDATE goalslides SET generated_visualization_script = ? WHERE id = ?',
                (visualization_script, entry_id)
            )
            conn.commit()

            # Fetch the updated entry
            cursor.execute('SELECT * FROM goalslides WHERE id = ?', (entry_id,))
            updated_entry = Goalslide(*cursor.fetchone())

            # Return the updated visualization script section
            return Div(
                H4("Visualization Script:"),
                Pre(
                    visualization_script,
                    cls="visualization-script",
                    id=f"visualization-script-text-{entry_id}"
                ),
                Button(
                    "Play Visualization Script",
                    onclick=f"playVisualizationScript({entry_id})",
                    cls="play-visualization-button"
                ),
                Button(
                    "Regenerate Visualization Script",
                    onclick=f"generateVisualizationScript({entry_id})",
                    cls="generate-script-button"
                ),
                cls="visualization-script-container",
                id=f"visualization-script-container-{entry_id}"
            )
        except Exception as e:
            return Div(f"Error generating visualization script: {str(e)}", cls="error-message")
    else:
        return Div("Goalslide not found.", cls="error-message")


@rt("/journal/auto-title/{entry_id}", methods=["POST"])
def auto_title_journal(entry_id: int):
    entry = find_journal_entry(entry_id)
    if entry:
        content = entry.content
        # Construct the prompt for the OpenAI API
        prompt = f"Provide a concise and informative title for the following journal entry content:\n\n{content}\n\nTitle:"
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # or "gpt-4" if you have access
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            new_title = response.choices[0].message.content.replace('"', '')
            # new_title = response.choices[0].message['content'].strip()
            # Update the journal entry's title in the database
            cursor.execute('UPDATE journal_entries SET title = ? WHERE id = ?', (new_title, entry_id))
            conn.commit()
            # Fetch the updated entry
            updated_entry = find_journal_entry(entry_id)
            # Return the updated journal entry
            return render_journal_entry(updated_entry)
        except Exception as e:
            # Handle API errors
            return Div(f"Error generating title: {str(e)}")
    else:
        return Div("Journal entry not found.")


@rt("/journal/generate-tasks/{entry_id}", methods=["POST"])
def generate_tasks(entry_id: int):
    entry = find_journal_entry(entry_id)
    if entry:
        content = entry.content
        # Construct the messages for the assistant
        cursor.execute('SELECT DISTINCT category FROM tasks')
        existing_categories = [row[0] for row in cursor.fetchall() if row[0]]
        existing_categories_text = ', '.join(existing_categories) if existing_categories else 'None'

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that extracts tasks from a journal entry. For each task, provide a description, assign a priority level (1 for High, 2 for Medium, 3 for Low), assign a category (either from the existing categories or create a new one), and determine its urgency on a scale from 1 (least urgent) to 10 (most urgent)."
            },
            {
                "role": "user",
                "content": f"Journal Entry:\n{content}\n\nExisting categories: {existing_categories_text}\n\nExtract the tasks."
            }
        ]

        # Define the function schema
        functions = [
            {
                "name": "extract_tasks",
                "description": "Extracts tasks from text",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tasks": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "description": {
                                        "type": "string",
                                        "description": "Task description"
                                    },
                                    "priority": {
                                        "type": "integer",
                                        "description": "Task priority (1 for High, 2 for Medium, 3 for Low)"
                                    },
                                    "category": {
                                        "type": "string",
                                        "description": "Category that best fits the task; use an existing category or suggest a new one"
                                    },
                                    "urgency": {
                                        "type": "integer",
                                        "description": "Urgency of the task on a scale from 1 (least urgent) to 10 (most urgent)"
                                    }
                                },
                                "required": ["description", "priority", "category", "urgency"],
                            }
                        }
                    },
                    "required": ["tasks"],
                }
            }
        ]

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # or "gpt-4" if you have access
                messages=messages,
                functions=functions,
                function_call={"name": "extract_tasks"},
            )

            message = response.choices[0].message

            if message.function_call:
                # Extract the arguments
                function_args = json.loads(message.function_call.arguments)
                tasks_data = function_args.get("tasks", [])

                new_tasks = []
                for task_data in tasks_data:
                    description = task_data.get("description")
                    priority = task_data.get("priority")
                    category = task_data.get("category")
                    urgency = int(task_data.get("urgency"))
                    if priority not in [1, 2, 3]:
                        priority = 4  # Default to 'None' if invalid
                    # Insert into database
                    cursor.execute(
                        'INSERT INTO tasks (description, priority, category, urgency) VALUES (?, ?, ?, ?)',
                        (description, priority, category, urgency)
                    )
                    conn.commit()
                    new_task_id = cursor.lastrowid
                    cursor.execute(
                        'SELECT id, description, priority, status, category, urgency FROM tasks WHERE id = ?',
                        (new_task_id,)
                    )
                    new_task = dict(
                        zip(['id', 'description', 'priority', 'status', 'category', 'urgency'], cursor.fetchone())
                    )
                    new_tasks.append(new_task)

                # Render a preview of the new tasks
                preview = Div(
                    H3("Generated Tasks"),
                    *(render_task(task) for task in new_tasks),
                    cls="tasks-preview"
                )

                return preview
            else:
                return Div("No tasks were generated.", cls="no-tasks")
        except Exception as e:
            return Div(f"Error generating tasks: {str(e)}", cls="error-message")
    else:
        return Div("Journal entry not found.", cls="error-message")


# Route to display the mood log
@rt("/mood", methods=["GET"])
def get_mood_log():
    cursor.execute('SELECT * FROM mood_entries ORDER BY date DESC')
    entries = [render_mood_entry(MoodEntry(id=row[0], mood=row[1], note=row[2], date=row[3]))
               for row in cursor.fetchall()]

    add_form = Form(
        H3("Log Your Mood"),
        Select(
            Option("Happy", value="Happy"),
            Option("Sad", value="Sad"),
            Option("Excited", value="Excited"),
            Option("Anxious", value="Anxious"),
            Option("Calm", value="Calm"),
            Option("Angry", value="Angry"),
            Option("Frustrated", value="Frustrated"),
            Option("Hopeful", value="Hopeful"),
            Option("Content", value="Content"),
            Option("Confused", value="Confused"),
            Option("Motivated", value="Motivated"),
            Option("Lonely", value="Lonely"),
            Option("Grateful", value="Grateful"),
            Option("Bored", value="Bored"),
            Option("Stressed", value="Stressed"),
            Option("Overwhelmed", value="Overwhelmed"),
            Option("Disappointed", value="Disappointed"),
            Option("Energized", value="Energized"),
            Option("Relaxed", value="Relaxed"),
            Option("Nervous", value="Nervous"),
            Option("Guilty", value="Guilty"),
            Option("Relieved", value="Relieved"),
            Option("Apathetic", value="Apathetic"),
            Option("Inspired", value="Inspired"),
            Option("Optimistic", value="Optimistic"),
            Option("Pessimistic", value="Pessimistic"),
            Option("Insecure", value="Insecure"),
            Option("Surprised", value="Surprised"),
            Option("Curious", value="Curious"),
            Option("Disheartened", value="Disheartened"),
            Option("Empowered", value="Empowered"),
            Option("Jealous", value="Jealous"),
            Option("Ashamed", value="Ashamed"),
            Option("Proud", value="Proud"),
            Option("Envious", value="Envious"),
            Option("Disgusted", value="Disgusted"),
            Option("Lethargic", value="Lethargic"),
            Option("Playful", value="Playful"),
            Option("Worried", value="Worried"),
            Option("Ecstatic", value="Ecstatic"),
            Option("Resentful", value="Resentful"),
            Option("Miserable", value="Miserable"),
            Option("Trusting", value="Trusting"),
            Option("Rejected", value="Rejected"),
            Option("Affectionate", value="Affectionate"),
            Option("Vulnerable", value="Vulnerable"),
            Option("Disconnected", value="Disconnected"),
            Option("Accepted", value="Accepted"),
            Option("Secure", value="Secure"),
            Option("Rejected", value="Rejected"),
            Option("Embarrassed", value="Embarrassed"),
            Option("Indifferent", value="Indifferent"),
            Option("Frightened", value="Frightened"),
            Option("Appreciative", value="Appreciative"),
            name="mood"
        ),
        Textarea(name="note", placeholder="Add a note (optional)"),
        Button("Log Mood", cls="add-button"),
        hx_post="/mood/add",
        hx_swap="afterbegin",
        hx_target="#mood-entries-list",
        cls="add-mood-form"
    )

    layout = Div(
        create_tabs("mood log"),
        Div(
            Div(add_form, cls="sidebar"),
            Div(Div(*entries, id="mood-entries-list"), cls="main-content"),
            cls="content-layout"
        ),
        cls="layout"
    )

    return layout, Link(rel="stylesheet", href="/static/styles.css")


# Route to add a new journal entry
@rt("/journal/add", methods=["POST"])
def post_journal(title: str, content: str):
    date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('INSERT INTO journal_entries (title, content, date) VALUES (?, ?, ?)', (title, content, date))
    conn.commit()
    new_entry_id = cursor.lastrowid
    new_entry = find_journal_entry(new_entry_id)
    return render_journal_entry(new_entry)


# Route to add a new mood entry
@rt("/mood/add", methods=["POST"])
def post_mood(mood: str, note: str):
    date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('INSERT INTO mood_entries (mood, note, date) VALUES (?, ?, ?)', (mood, note, date))
    conn.commit()
    new_entry_id = cursor.lastrowid
    new_entry = find_mood_entry(new_entry_id)
    return render_mood_entry(new_entry)


# Route to delete a journal entry
@rt("/journal/delete/{entry_id}", methods=["DELETE"])
def delete_journal(entry_id: int):
    cursor.execute('DELETE FROM journal_entries WHERE id = ?', (entry_id,))
    conn.commit()
    return ""


# Route to delete a mood entry
@rt("/mood/delete/{entry_id}", methods=["DELETE"])
def delete_mood(entry_id: int):
    cursor.execute('DELETE FROM mood_entries WHERE id = ?', (entry_id,))
    conn.commit()
    return ""


# Route to display the edit form for a journal entry
@rt("/journal/edit/{entry_id}", methods=["GET"])
def get_journal_edit(entry_id: int):
    entry = find_journal_entry(entry_id)
    if entry:
        edit_form = Form(
            Input(name="title", value=entry.title),
            Textarea(name="content", value=entry.content),
            Button("Save", cls="save-button"),
            hx_put=f"/journal/edit/{entry.id}",
            hx_swap="outerHTML",
            cls="edit-entry-form"
        )
        return Div(edit_form, id=f"entry-{entry.id}")


# Route to display the edit form for a mood entry
@rt("/mood/edit/{entry_id}", methods=["GET"])
def get_mood_edit(entry_id: int):
    entry = find_mood_entry(entry_id)
    if entry:
        edit_form = Form(
            Select(
                Option("Happy", value="Happy", selected=entry.mood == "Happy"),
                Option("Sad", value="Sad", selected=entry.mood == "Sad"),
                Option("Excited", value="Excited", selected=entry.mood == "Excited"),
                Option("Anxious", value="Anxious", selected=entry.mood == "Anxious"),
                Option("Calm", value="Calm", selected=entry.mood == "Calm"),
                name="mood"
            ),
            Textarea(name="note", value=entry.note),
            Button("Save", cls="save-button"),
            hx_put=f"/mood/edit/{entry.id}",
            hx_swap="outerHTML",
            cls="edit-mood-form"
        )
        return Div(edit_form, id=f"mood-entry-{entry.id}")


# Route to update a journal entry
@rt("/journal/edit/{entry_id}", methods=["PUT"])
def put_journal(entry_id: int, title: str, content: str):
    cursor.execute('UPDATE journal_entries SET title = ?, content = ? WHERE id = ?', (title, content, entry_id))
    conn.commit()
    updated_entry = find_journal_entry(entry_id)
    return render_journal_entry(updated_entry)


@rt("/journal/generate-cbt/{entry_id}", methods=["POST"])
def generate_cbt_entries(entry_id: int):
    entry = find_journal_entry(entry_id)
    if entry:
        content = entry.content
        # Construct the messages for the assistant
        messages = [
            {"role": "system",
             "content": "You are a helpful assistant that extracts Cognitive Behavioral Therapy (CBT) entries from a journal entry. For each negative thought, identify the cognitive distortion and provide a rational rebuttal."},
            {"role": "user", "content": f"Journal Entry:\n{content}\n\nExtract the CBT entries."}
        ]

        # Define the function schema
        functions = [
            {
                "name": "extract_cbt_entries",
                "description": "Extracts CBT entries from text",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cbt_entries": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "negative_thought": {"type": "string",
                                                         "description": "A negative thought from the journal entry"},
                                    "cognitive_distortion": {"type": "string",
                                                             "description": "The cognitive distortion associated with the negative thought"},
                                    "rational_rebuttal": {"type": "string",
                                                          "description": "A rational rebuttal to the negative thought"},
                                },
                                "required": ["negative_thought", "cognitive_distortion", "rational_rebuttal"],
                            }
                        }
                    },
                    "required": ["cbt_entries"],
                }
            }
        ]

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # Or "gpt-4-0613" if available
                messages=messages,
                functions=functions,
                function_call={"name": "extract_cbt_entries"},
            )
            message = response.choices[0].message

            if message.function_call:
                # Extract the arguments
                function_args = json.loads(message.function_call.arguments)
                cbt_entries_data = function_args.get("cbt_entries", [])

                new_entries = []
                for cbt_data in cbt_entries_data:
                    negative_thought = cbt_data.get("negative_thought")
                    cognitive_distortion = cbt_data.get("cognitive_distortion")
                    rational_rebuttal = cbt_data.get("rational_rebuttal")
                    date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    # Insert into database
                    cursor.execute(
                        'INSERT INTO cbt_entries (negative_thought, cognitive_distortion, rational_rebuttal, date) VALUES (?, ?, ?, ?)',
                        (negative_thought, cognitive_distortion, rational_rebuttal, date))
                    conn.commit()
                    new_entry_id = cursor.lastrowid
                    new_entry = CBTEntry(id=new_entry_id, negative_thought=negative_thought,
                                         cognitive_distortion=cognitive_distortion, rational_rebuttal=rational_rebuttal,
                                         date=date)
                    new_entries.append(new_entry)

                # Render a preview of the new entries
                preview = Div(
                    H3("Generated CBT Entries"),
                    *(render_cbt_entry(entry) for entry in new_entries),
                    cls="cbt-preview"
                )

                return preview
            else:
                return Div("No CBT entries were generated.", cls="no-cbt-entries")
        except Exception as e:
            return Div(f"Error generating CBT entries: {str(e)}", cls="error-message")
    else:
        return Div("Journal entry not found.", cls="error-message")


# Route to update a mood entry
@rt("/mood/edit/{entry_id}", methods=["PUT"])
def put_mood(entry_id: int, mood: str, note: str):
    cursor.execute('UPDATE mood_entries SET mood = ?, note = ? WHERE id = ?', (mood, note, entry_id))
    conn.commit()
    updated_entry = find_mood_entry(entry_id)
    return render_mood_entry(updated_entry)


# Route to handle audio transcription
@rt("/transcribe", methods=["POST"])
async def post(audio_data: UploadFile):
    temp_file_path = None
    try:
        logger.info("1. Starting transcription process")

        if not audio_data.filename.lower().endswith('.wav'):
            logger.info(f"2. Invalid file format: {audio_data.filename}")
            return HTMLResponse(content="Invalid file format. Please upload a WAV file.", status_code=400)

        logger.info("3. Reading audio data")
        content = await audio_data.read()
        if len(content) == 0:
            logger.info("4. Empty audio file received")
            return HTMLResponse(content="Empty audio file received", status_code=400)

        logger.info(f"5. Audio data size: {len(content)} bytes")

        logger.info("6. Creating temporary file")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file_path = temp_file.name
            logger.info(f"7. Temporary file created: {temp_file_path}")
            temp_file.write(content)
            logger.info("8. Content written to temporary file")

        logger.info("9. Starting transcription")
        transcribed_text = transcribe_audio(client, temp_file_path)
        if not transcribed_text:
            logger.info("10. Transcription failed or returned empty")
            return HTMLResponse(content="Transcription failed or returned empty", status_code=500)

        logger.info(f"11. Transcription successful. Text length: {len(transcribed_text)}")

        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        logger.info("12. Inserting into database")
        cursor.execute('INSERT INTO journal_entries (title, content, date) VALUES (?, ?, ?)',
                       ("Voice Journal Entry", transcribed_text, date))
        conn.commit()
        new_entry_id = cursor.lastrowid

        if not new_entry_id:
            logger.info("13. Failed to insert new entry into database")
            return HTMLResponse(content="Failed to insert new entry into database", status_code=500)

        logger.info(f"14. New entry inserted. ID: {new_entry_id}")

        new_entry = find_journal_entry(new_entry_id)
        logger.info("15. Rendering journal entry")
        return render_journal_entry(new_entry)

    except Exception as e:
        logger.error(f"ERROR: {str(e)}", exc_info=True)
        logger.error(f"Error Type: {type(e).__name__}")
        logger.error(f"Error Args: {e.args}")
        logger.error(f"Error occurred at step: {locals().get('step', 'Unknown')}")
        error_message = f"Error: {str(e)}\n"
        error_message += f"Error Type: {type(e).__name__}\n"
        error_message += f"Error Args: {e.args}\n"
        return HTMLResponse(content=error_message, status_code=500)
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            logger.info(f"16. Removing temporary file: {temp_file_path}")
            os.remove(temp_file_path)
        logger.info("17. Transcription process completed")


# JavaScript for audio recording
audio_recording_js = """
let mediaRecorder;
let audioChunks = [];

document.getElementById('start-recording-button').addEventListener('click', startRecording);
document.getElementById('stop-recording-button').addEventListener('click', stopRecording);

async function startRecording() {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);

    mediaRecorder.ondataavailable = (event) => {
        audioChunks.push(event.data);
    };

    mediaRecorder.onstop = sendAudioToServer;

    mediaRecorder.start();
    document.getElementById('start-recording-button').style.display = 'none';
    document.getElementById('stop-recording-button').style.display = 'inline';
    document.getElementById('recording-status').textContent = 'Recording...';
}

function stopRecording() {
    mediaRecorder.stop();
    document.getElementById('start-recording-button').style.display = 'inline';
    document.getElementById('stop-recording-button').style.display = 'none';
    document.getElementById('recording-status').textContent = 'Processing...';
}

function sendAudioToServer() {
    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
    const formData = new FormData();
    formData.append('audio_data', audioBlob, 'recording.wav');

    // Determine the correct endpoint based on the current URL
    let endpoint = '/transcribe';  // Default to '/transcribe'
    if (window.location.pathname.startsWith('/notes')) {
        endpoint = '/notes/transcribe';
    }

    fetch(endpoint, {
        method: 'POST',
        body: formData
    })
    .then(response => response.text())
    .then(html => {
        document.querySelector('body').insertAdjacentHTML('beforeend', html);
        document.getElementById('recording-status').textContent = '';
        audioChunks = [];
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('recording-status').textContent = 'Error processing audio';
    });
}
"""


@rt("/challenges/progress/update/{progress_id}", methods=["POST"])
def update_challenge_progress(progress_id: int, completed: str):
    # Convert 'true'/'false' string to integer 1/0
    completed_int = 1 if completed.lower() == 'true' else 0
    cursor.execute('UPDATE challenge_progress SET completed = ? WHERE id = ?', (completed_int, progress_id))
    conn.commit()

    # Fetch the challenge ID to re-render the challenge
    cursor.execute('SELECT challenge_id FROM challenge_progress WHERE id = ?', (progress_id,))
    challenge_id = cursor.fetchone()[0]

    # Fetch the updated challenge
    cursor.execute('SELECT * FROM challenges WHERE id = ?', (challenge_id,))
    challenge = Challenge(*cursor.fetchone())

    return render_challenge(challenge)

@rt("/challenges/transformations", methods=["GET"])
def get_transformations():
    # Fetch transformations from the database
    cursor.execute('SELECT * FROM transformations ORDER BY date_added DESC')
    transformations = [Transformation(*row) for row in cursor.fetchall()]

    transformation_entries = [render_transformation(transformation) for transformation in transformations]

    add_form = Form(
        H3("Add New Transformation"),
        Textarea(name="description", placeholder="Describe your transformation"),
        Button("Add Transformation", cls="add-button"),
        hx_post="/challenges/transformations/add",
        hx_swap="afterbegin",
        hx_target="#transformations-list",
        cls="add-transformation-form"
    )

    # Add the 'Generate Transformation' button
    generate_button = Button(
        "Generate Transformation",
        hx_post="/challenges/transformations/generate",
        hx_target="#transformations-list",
        hx_swap="afterbegin",
        cls="generate-transformation-button"
    )

    layout = Div(
        create_tabs("Challenges"),
        create_challenges_subtabs("Transformations"),
        Div(
            Div(add_form, generate_button, cls="sidebar"),
            Div(
                Div(*transformation_entries, id="transformations-list"),
                cls="main-content"
            ),
            cls="content-layout"
        ),
        cls="layout"
    )

    return layout, Link(rel="stylesheet", href="/static/styles.css")
@rt("/challenges/transformations/generate", methods=["POST"])
def generate_transformation():
    # Fetch data from the last month
    date_one_month_ago = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d %H:%M:%S")

    # Fetch journal entries
    cursor.execute('SELECT content FROM journal_entries WHERE date >= ?', (date_one_month_ago,))
    journal_entries = [row[0] for row in cursor.fetchall()]

    # Fetch mood entries
    cursor.execute('SELECT mood, note FROM mood_entries WHERE date >= ?', (date_one_month_ago,))
    mood_entries = [{'mood': row[0], 'note': row[1]} for row in cursor.fetchall()]

    # Fetch tasks
    cursor.execute('SELECT description, priority, status FROM tasks WHERE status != "completed"')
    tasks = [dict(description=row[0], priority=row[1], status=row[2]) for row in cursor.fetchall()]

    # Fetch EMDR sessions
    cursor.execute('SELECT feelings, memory, body_sensations FROM emdr_sessions WHERE date >= ?', (date_one_month_ago,))
    emdr_sessions = [{'feelings': row[0], 'memory': row[1], 'body_sensations': row[2]} for row in cursor.fetchall()]

    # Prepare the data for the AI
    data = {
        'journal_entries': journal_entries,
        'mood_entries': mood_entries,
        'tasks': tasks,
        'emdr_sessions': emdr_sessions,
    }

    # Construct the prompt
    prompt = f"""
Over the past month, I've been recording various entries in my personal journal app. Below is the data collected from different aspects of my life:

Journal Entries:
{chr(10).join(journal_entries)}

Mood Entries:
{chr(10).join([f"Mood: {entry['mood']}, Note: {entry['note']}" for entry in mood_entries])}

Current Tasks:
{chr(10).join([f"Description: {task['description']}, Priority: {task['priority']}, Status: {task['status']}" for task in tasks])}

EMDR Sessions:
{chr(10).join([f"Feelings: {session['feelings']}, Memory: {session['memory']}, Body Sensations: {session['body_sensations']}" for session in emdr_sessions])}

Based on this information, please suggest a personal transformation that I could undertake. Consider unique, inspiring, and possibly unconventional experiences that could contribute to personal growth and well-being.

Your suggestion should be a concise description of the transformation and how it relates to the patterns or themes observed in the data.

Transformation:
"""

    # Call the AI to generate the transformation
    try:
        response = anthropic_client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        print(response.content)
        return response.content
        # response = client.chat.completions.create(
        #     model="o1-preview",  # Use the appropriate model
        #     messages=[
        #         {"role": "user", "content": prompt}
        #     ]
        #
        #
        # )
        transformation_text = response.choices[0].message.content.strip()

        # Save the transformation to the database
        date_added = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute('INSERT INTO transformations (description, date_added) VALUES (?, ?)', (transformation_text, date_added))
        conn.commit()
        new_transformation_id = cursor.lastrowid

        new_transformation = Transformation(id=new_transformation_id, description=transformation_text, date_added=date_added)

        # Render the new transformation and insert it into the page
        return render_transformation(new_transformation)

    except Exception as e:
        return Div(f"Error generating transformation: {str(e)}", cls="error-message")

@rt("/challenges/transformations/delete/{transformation_id}", methods=["DELETE"])
def delete_transformation(transformation_id: int):
    cursor.execute('DELETE FROM transformations WHERE id = ?', (transformation_id,))
    conn.commit()
    return ""

@rt("/challenges/transformations/edit/{transformation_id}", methods=["GET"])
def get_transformation_edit(transformation_id: int):
    cursor.execute('SELECT * FROM transformations WHERE id = ?', (transformation_id,))
    row = cursor.fetchone()
    if row:
        transformation = Transformation(*row)
        edit_form = Form(
            Textarea(name="description", value=transformation.description),
            Button("Save", cls="save-button"),
            hx_put=f"/challenges/transformations/edit/{transformation.id}",
            hx_swap="outerHTML",
            cls="edit-transformation-form"
        )
        return Div(edit_form, id=f"transformation-{transformation.id}")
    else:
        return Div("Transformation not found.", cls="error-message")

@rt("/challenges/transformations/edit/{transformation_id}", methods=["PUT"])
def update_transformation(transformation_id: int, description: str):
    cursor.execute('UPDATE transformations SET description = ? WHERE id = ?', (description, transformation_id))
    conn.commit()
    cursor.execute('SELECT * FROM transformations WHERE id = ?', (transformation_id,))
    updated_row = cursor.fetchone()
    if updated_row:
        updated_transformation = Transformation(*updated_row)
        return render_transformation(updated_transformation)
    else:
        return Div("Transformation not found.", cls="error-message")

def render_transformation(transformation):
    return Div(
        P(transformation.description),
        P(f"Date Added: {transformation.date_added}", cls="transformation-date"),
        Div(
            A("Edit", href=f"/challenges/transformations/edit/{transformation.id}", cls="edit-link"),
            A("Delete", hx_delete=f"/challenges/transformations/delete/{transformation.id}", hx_swap="outerHTML",
              cls="delete-link"),
            cls="transformation-actions"
        ),
        cls="transformation-entry",
        id=f"transformation-{transformation.id}"
    )
@rt("/challenges/transformations/add", methods=["POST"])
def add_transformation(description: str):
    date_added = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('INSERT INTO transformations (description, date_added) VALUES (?, ?)', (description, date_added))
    conn.commit()
    new_transformation_id = cursor.lastrowid
    new_transformation = Transformation(id=new_transformation_id, description=description, date_added=date_added)
    return render_transformation(new_transformation)

def create_challenges_subtabs(active_subtab):
    challenges_subtabs = [
        ("All Challenges", "/challenges"),
        ("3-days", "/challenges/3-days"),
        ("30-days", "/challenges/30-days"),
        ("Challenge Ideas", "/challenges/ideas"),
        ("Transformations", "/challenges/transformations"),  # Added "Transformations" subtab
    ]
    subtab_elements = [
        A(
            name,
            href=url,
            cls=f"subtab {'active' if name.lower() == active_subtab.lower() else ''}"
        )
        for name, url in challenges_subtabs
    ]
    return Div(*subtab_elements, cls="subtab-navigation")


@rt("/challenges/30-days", methods=["GET"])
def get_30day_challenges():
    # Fetch 30-day challenges from the database
    cursor.execute('SELECT * FROM challenges WHERE duration = 30 ORDER BY start_date DESC')
    challenges = [Challenge(*row) for row in cursor.fetchall()]

    challenge_entries = [render_challenge(challenge) for challenge in challenges]

    add_form = Form(
        H3("Create New 30-Day Challenge"),
        Input(name="title", placeholder="Challenge Title"),
        Textarea(name="description", placeholder="Challenge Description"),
        Button("Add Challenge", cls="add-button"),
        hx_post="/challenges/30-days/add",
        hx_swap="afterbegin",
        hx_target="#challenges-list",
        cls="add-challenge-form"
    )

    layout = Div(
        create_tabs("Challenges"),
        create_challenges_subtabs("30-days"),
        Div(
            Div(add_form, cls="sidebar"),
            Div(Div(*challenge_entries, id="challenges-list"), cls="main-content"),
            cls="content-layout"
        ),
        cls="layout"
    )

    return layout, Link(rel="stylesheet", href="/static/styles.css"), Script(src="/static/challenges.js")


@rt("/challenges/30-days/add", methods=["POST"])
def post_30day_challenge(title: str, description: str):
    duration = 30
    start_date = datetime.now().strftime("%Y-%m-%d")
    end_date = (datetime.now() + timedelta(days=duration - 1)).strftime("%Y-%m-%d")
    reminders = 0  # Adjust as needed

    cursor.execute('''
        INSERT INTO challenges (title, description, start_date, end_date, duration, reminders_enabled)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (title, description, start_date, end_date, duration, reminders))
    conn.commit()
    challenge_id = cursor.lastrowid

    # Initialize progress entries for the challenge duration
    for i in range(duration):
        progress_date = (datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d")
        cursor.execute('''
            INSERT INTO challenge_progress (challenge_id, progress_date)
            VALUES (?, ?)
        ''', (challenge_id, progress_date))
    conn.commit()

    new_challenge = Challenge(
        id=challenge_id,
        title=title,
        description=description,
        start_date=start_date,
        end_date=end_date,
        duration=duration,
        reminders_enabled=reminders
    )
    return render_challenge(new_challenge)


@rt("/challenges", methods=["GET"])
def get_challenges():
    cursor.execute('SELECT * FROM challenges ORDER BY start_date DESC')
    challenges = [Challenge(*row) for row in cursor.fetchall()]

    challenge_entries = [render_challenge(challenge) for challenge in challenges]

    add_form = Form(
        H3("Create New Challenge"),
        Input(name="title", placeholder="Challenge Title"),
        Textarea(name="description", placeholder="Challenge Description"),
        Label(
            Input(type="checkbox", name="reminders_enabled"),
            " Enable Daily Reminders"
        ),
        Button("Add Challenge", cls="add-button"),
        hx_post="/challenges/add",
        hx_swap="afterbegin",
        hx_target="#challenges-list",
        cls="add-challenge-form"
    )

    layout = Div(
        create_tabs("Challenges"),
        create_challenges_subtabs("All Challenges"),
        Div(
            Div(add_form, cls="sidebar"),
            Div(Div(*challenge_entries, id="challenges-list"), cls="main-content"),
            cls="content-layout"
        ),
        cls="layout"
    )

    return layout, Link(rel="stylesheet", href="/static/styles.css"), Script(src="/static/challenges.js")


@rt("/challenges/ideas", methods=["GET"])
def get_challenge_ideas():
    # Fetch challenge ideas from the database
    cursor.execute('SELECT * FROM challenge_ideas ORDER BY date_added DESC')
    ideas = [ChallengeIdea(*row) for row in cursor.fetchall()]

    idea_entries = [render_challenge_idea(idea) for idea in ideas]

    add_form = Form(
        H3("Add New Challenge Idea"),
        Input(name="title", placeholder="Idea Title"),
        Textarea(name="description", placeholder="Idea Description"),
        Button("Add Idea", cls="add-button"),
        hx_post="/challenges/ideas/add",
        hx_swap="afterbegin",
        hx_target="#challenge-ideas-list",
        cls="add-idea-form"
    )

    layout = Div(
        create_tabs("Challenges"),
        create_challenges_subtabs("Challenge Ideas"),
        Div(
            Div(add_form, cls="sidebar"),
            Div(Div(*idea_entries, id="challenge-ideas-list"), cls="main-content"),
            cls="content-layout"
        ),
        cls="layout"
    )

    return layout, Link(rel="stylesheet", href="/static/styles.css")


def render_challenge_idea(idea):
    return Div(
        H3(idea.title),
        P(idea.description),
        P(f"Date Added: {idea.date_added}", cls="idea-date"),
        Div(
            A("Edit", href=f"/challenges/ideas/edit/{idea.id}", cls="edit-link"),
            A("Delete", hx_delete=f"/challenges/ideas/delete/{idea.id}", hx_swap="outerHTML", cls="delete-link"),
            A("Create Challenge", hx_post=f"/challenges/ideas/create/{idea.id}", cls="create-challenge-link"),
            cls="idea-actions"
        ),
        cls="challenge-idea-entry",
        id=f"challenge-idea-{idea.id}"
    )


@rt("/challenges/ideas/add", methods=["POST"])
def add_challenge_idea(title: str, description: str):
    date_added = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('INSERT INTO challenge_ideas (title, description, date_added) VALUES (?, ?, ?)',
                   (title, description, date_added))
    conn.commit()
    new_idea_id = cursor.lastrowid
    new_idea = ChallengeIdea(id=new_idea_id, title=title, description=description, date_added=date_added)
    return render_challenge_idea(new_idea)


@rt("/challenges/ideas/delete/{idea_id}", methods=["DELETE"])
def delete_challenge_idea(idea_id: int):
    cursor.execute('DELETE FROM challenge_ideas WHERE id = ?', (idea_id,))
    conn.commit()
    return ""


@rt("/challenges/ideas/edit/{idea_id}", methods=["GET"])
def get_challenge_idea_edit(idea_id: int):
    cursor.execute('SELECT * FROM challenge_ideas WHERE id = ?', (idea_id,))
    row = cursor.fetchone()
    if row:
        idea = ChallengeIdea(*row)
        edit_form = Form(
            Input(name="title", value=idea.title),
            Textarea(name="description", value=idea.description),
            Button("Save", cls="save-button"),
            hx_put=f"/challenges/ideas/edit/{idea.id}",
            hx_swap="outerHTML",
            cls="edit-idea-form"
        )
        return Div(edit_form, id=f"challenge-idea-{idea.id}")
    else:
        return Div("Challenge idea not found.", cls="error-message")


@rt("/challenges/ideas/edit/{idea_id}", methods=["PUT"])
def update_challenge_idea(idea_id: int, title: str, description: str):
    cursor.execute('UPDATE challenge_ideas SET title = ?, description = ? WHERE id = ?',
                   (title, description, idea_id))
    conn.commit()
    updated_idea = ChallengeIdea(id=idea_id, title=title, description=description,
                                 date_added=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    return render_challenge_idea(updated_idea)


@rt("/challenges/ideas/create/{idea_id}", methods=["POST"])
def create_challenge_from_idea(idea_id: int):
    # Fetch the idea from the database
    cursor.execute('SELECT * FROM challenge_ideas WHERE id = ?', (idea_id,))
    row = cursor.fetchone()
    if row:
        idea = ChallengeIdea(*row)
        # Create a new challenge with default values or prompt the user for additional details
        duration = 30  # Default duration, you can adjust as needed
        start_date = datetime.now().strftime("%Y-%m-%d")
        end_date = (datetime.now() + timedelta(days=duration - 1)).strftime("%Y-%m-%d")
        reminders = 0  # Adjust as needed

        cursor.execute('''
            INSERT INTO challenges (title, description, start_date, end_date, duration, reminders_enabled)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (idea.title, idea.description, start_date, end_date, duration, reminders))
        conn.commit()
        challenge_id = cursor.lastrowid

        # Initialize progress entries
        for i in range(duration):
            progress_date = (datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d")
            cursor.execute('''
                INSERT INTO challenge_progress (challenge_id, progress_date)
                VALUES (?, ?)
            ''', (challenge_id, progress_date))
        conn.commit()

        return Div("Challenge created successfully.", cls="success-message")
    else:
        return Div("Challenge idea not found.", cls="error-message")


@rt("/challenges/3-days", methods=["GET"])
def get_3day_challenges():
    # Fetch 3-day challenges from the database
    cursor.execute('SELECT * FROM challenges WHERE duration = 3 ORDER BY start_date DESC')
    challenges = [Challenge(*row) for row in cursor.fetchall()]

    challenge_entries = [render_challenge(challenge) for challenge in challenges]

    add_form = Form(
        H3("Create New 3-Day Challenge"),
        Input(name="title", placeholder="Challenge Title"),
        Textarea(name="description", placeholder="Challenge Description"),
        Button("Add Challenge", cls="add-button"),
        hx_post="/challenges/3-days/add",
        hx_swap="afterbegin",
        hx_target="#challenges-list",
        cls="add-challenge-form"
    )

    layout = Div(
        create_tabs("Challenges"),
        create_challenges_subtabs("3-days"),
        Div(
            Div(add_form, cls="sidebar"),
            Div(Div(*challenge_entries, id="challenges-list"), cls="main-content"),
            cls="content-layout"
        ),
        cls="layout"
    )

    return layout, Link(rel="stylesheet", href="/static/styles.css"), Script(src="/static/challenges.js")


@rt("/challenges/3-days/add", methods=["POST"])
def post_3day_challenge(title: str, description: str):
    duration = 3
    start_date = datetime.now().strftime("%Y-%m-%d")
    end_date = (datetime.now() + timedelta(days=duration - 1)).strftime("%Y-%m-%d")
    reminders = 0  # Adjust as needed

    cursor.execute('''
        INSERT INTO challenges (title, description, start_date, end_date, duration, reminders_enabled)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (title, description, start_date, end_date, duration, reminders))
    conn.commit()
    challenge_id = cursor.lastrowid

    # Initialize progress entries for the challenge duration
    for i in range(duration):
        progress_date = (datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d")
        cursor.execute('''
            INSERT INTO challenge_progress (challenge_id, progress_date)
            VALUES (?, ?)
        ''', (challenge_id, progress_date))
    conn.commit()

    new_challenge = Challenge(
        id=challenge_id,
        title=title,
        description=description,
        start_date=start_date,
        end_date=end_date,
        duration=duration,
        reminders_enabled=reminders
    )
    return render_challenge(new_challenge)


@rt("/challenges/add", methods=["POST"])
def post_challenge(title: str, description: str, reminders_enabled: Optional[str] = None):
    duration = 30
    start_date = datetime.now().strftime("%Y-%m-%d")
    end_date = (datetime.now() + timedelta(days=duration - 1)).strftime("%Y-%m-%d")
    reminders = 1 if reminders_enabled == 'on' else 0

    cursor.execute('''
        INSERT INTO challenges (title, description, start_date, end_date, duration, reminders_enabled)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (title, description, start_date, end_date, duration, reminders))
    conn.commit()
    challenge_id = cursor.lastrowid

    # Initialize progress entries
    for i in range(duration):
        progress_date = (datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d")
        cursor.execute('''
            INSERT INTO challenge_progress (challenge_id, progress_date)
            VALUES (?, ?)
        ''', (challenge_id, progress_date))
    conn.commit()

    new_challenge = Challenge(
        id=challenge_id,
        title=title,
        description=description,
        start_date=start_date,
        end_date=end_date,
        duration=duration,
        reminders_enabled=reminders
    )
    return render_challenge(new_challenge)


def render_challenge(challenge):
    # Fetch progress entries
    cursor.execute('''
        SELECT * FROM challenge_progress
        WHERE challenge_id = ? ORDER BY progress_date ASC
    ''', (challenge.id,))
    progress_entries = [ChallengeProgress(*row) for row in cursor.fetchall()]

    # Calculate completion percentage
    total_days = len(progress_entries)
    completed_days = sum(entry.completed for entry in progress_entries)
    completion_percentage = int((completed_days / total_days) * 100)

    # Progress bar
    progress_bar = Div(
        Div(style=f"width: {completion_percentage}%;", cls="progress-bar-fill"),
        cls="progress-bar"
    )

    # Progress checkboxes
    progress_checks = Div(
        *[Div(
            Input(
                type="checkbox",
                **({"checked": "checked"} if entry.completed else {}),
                onchange=f"updateProgress({entry.id}, {challenge.id}, this.checked)"
            ),
            Span(entry.progress_date),
            cls="progress-check-item"
        ) for entry in progress_entries],
        cls="progress-checks"
    )

    return Div(
        H3(challenge.title),
        P(challenge.description),
        P(f"Start Date: {challenge.start_date}"),
        P(f"End Date: {challenge.end_date}"),
        progress_bar,
        progress_checks,
        Div(
            A("Edit", href=f"/challenges/edit/{challenge.id}", cls="edit-link"),
            A("Delete", hx_delete=f"/challenges/delete/{challenge.id}", hx_swap="outerHTML", cls="delete-link"),
            cls="entry-actions"
        ),
        cls="challenge-entry",
        id=f"challenge-{challenge.id}"
    )


@rt("/challenges/delete/{challenge_id}", methods=["DELETE"])
def delete_challenge(challenge_id: int):
    cursor.execute('DELETE FROM challenges WHERE id = ?', (challenge_id,))
    cursor.execute('DELETE FROM challenge_progress WHERE challenge_id = ?', (challenge_id,))
    conn.commit()
    return ""


@rt("/emdr/transcribe-target", methods=["POST"])
async def transcribe_target_audio(audio_data: UploadFile):
    temp_file_path = None
    try:
        content = await audio_data.read()
        if len(content) == 0:
            return JSONResponse({"error": "Empty audio file received"}, status_code=400)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file_path = temp_file.name
            temp_file.write(content)

        # Transcribe audio using OpenAI Whisper API or other service
        transcribed_text = transcribe_audio(client, temp_file_path)
        if not transcribed_text:
            return JSONResponse({"error": "Transcription failed or returned empty"}, status_code=500)

        return JSONResponse({"transcription": transcribed_text})

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@rt("/emdr/suggest-targets", methods=["GET"])
def get_emdr_suggest_targets():
    # Fetch the last 20 journal entries
    cursor.execute('SELECT content FROM journal_entries ORDER BY date DESC LIMIT 20')
    journal_entries = [row[0] for row in cursor.fetchall()]

    if not journal_entries:
        return Div("No journal entries found to generate EMDR targets.", cls="error-message")

    # Construct the prompt
    prompt = construct_emdr_prompt(journal_entries)

    # Call the OpenAI API
    try:
        emdr_targets = generate_emdr_targets(prompt)
    except Exception as e:
        return Div(f"Error generating EMDR targets: {str(e)}", cls="error-message")

    # Render the targets
    return render_emdr_targets(emdr_targets)


def construct_emdr_prompt(journal_entries):
    entries_text = "\n\n".join(journal_entries)
    prompt = f"""
Based on the following journal entries, identify recurring themes, negative beliefs, traumatic events, or emotional triggers that could serve as targets for EMDR therapy. Provide a list of suggested EMDR targets with brief explanations.

Journal Entries:
{entries_text}

Please provide the EMDR targets in the following format:

1. [Target]: [Brief Explanation]
2. [Target]: [Brief Explanation]
...
"""
    return prompt


def generate_emdr_targets(prompt):
    response = client.chat.completions.create(
        model="o1-preview",
        messages=[{"role": "user", "content": prompt}]
    )
    emdr_targets = response.choices[0].message.content.strip()
    return emdr_targets


def render_emdr_targets(targets_text):
    return Div(
        H3("Suggested EMDR Targets"),
        Pre(targets_text, cls="emdr-targets-text"),
        cls="emdr-targets-container"
    )


@rt("/notes/transcribe", methods=["POST"])
async def transcribe_note_audio(audio_data: UploadFile):
    temp_file_path = None
    try:
        logger.info("Starting transcription process for notes")

        if not audio_data.filename.lower().endswith('.wav'):
            logger.info(f"Invalid file format: {audio_data.filename}")
            return HTMLResponse(content="Invalid file format. Please upload a WAV file.", status_code=400)

        content = await audio_data.read()
        if len(content) == 0:
            logger.info("Empty audio file received")
            return HTMLResponse(content="Empty audio file received", status_code=400)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file_path = temp_file.name
            temp_file.write(content)

        # Use OpenAI Whisper for transcription
        transcribed_text = transcribe_audio(client, temp_file_path)
        if not transcribed_text:
            logger.info("Transcription failed or returned empty")
            return HTMLResponse(content="Transcription failed or returned empty", status_code=500)

        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        cursor.execute('INSERT INTO notes_entries (title, content, date) VALUES (?, ?, ?)',
                       ("Voice Note Entry", transcribed_text, date))
        conn.commit()
        new_entry_id = cursor.lastrowid

        if not new_entry_id:
            logger.info("Failed to insert new entry into database")
            return HTMLResponse(content="Failed to insert new entry into database", status_code=500)

        new_entry = find_note_entry(new_entry_id)
        return render_note_entry(new_entry)

    except Exception as e:
        logger.error(f"ERROR: {str(e)}", exc_info=True)
        error_message = f"Error: {str(e)}\n"
        error_message += f"Error Type: {type(e).__name__}\n"
        error_message += f"Error Args: {e.args}\n"
        return HTMLResponse(content=error_message, status_code=500)
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

visualization_js_code = """
function playVisualizationScript(entryId) {
    const scriptElement = document.getElementById(`visualization-script-text-${entryId}`);
    if (scriptElement) {
        const text = scriptElement.textContent || scriptElement.innerText || "";
        if (text) {
            // Use the Web Speech API to read the text aloud
            const utterance = new SpeechSynthesisUtterance(text);
            // Customize voice properties if desired
            utterance.rate = 1;  // Speed (0.1 to 10)
            utterance.pitch = 1; // Pitch (0 to 2)

            // Optional: Choose a specific voice
            // const voices = window.speechSynthesis.getVoices();
            // utterance.voice = voices.find(voice => voice.name === 'Google US English');

            window.speechSynthesis.speak(utterance);
        } else {
            alert("Visualization script is empty.");
        }
    } else {
        alert("Visualization script not found.");
    }
}
"""

def emdr_tool():
    js_code = """
    let dotX, dotY, dotRadius, canvasWidth, canvasHeight, direction, dotSpeed;
    let clickRightSound, clickLeftSound;
    let isRunning = false;
    let startButton;
    let timerSelect, timerDisplay;
    let timerDuration = 0;
    let timeRemaining = 0;
    let timerInterval;

    function setup() {
        const canvas = document.getElementById('emdrCanvas');
        canvasWidth = canvas.width = window.innerWidth;
        canvasHeight = canvas.height = window.innerHeight;
        dotRadius = 60;
        dotX = canvasWidth / 2;
        dotY = canvasHeight / 2;
        dotSpeed = 40;
        direction = 1;

        clickRightSound = new Audio('/static/click_right.wav');
        clickLeftSound = new Audio('/static/click_left.wav');

        // Preload sounds
        clickRightSound.load();
        clickLeftSound.load();

        window.addEventListener('resize', resizeCanvas);
        window.addEventListener('keydown', handleKeyPress);

        // Draw initial state
        drawStaticState();

        // Add start button
        startButton = document.createElement('button');
        startButton.textContent = 'Start EMDR';
        startButton.style.position = 'absolute';
        startButton.style.top = '20px';
        startButton.style.left = '20px';
        startButton.style.transition = 'opacity 0.3s ease-in-out';
        document.body.appendChild(startButton);

        startButton.addEventListener('click', toggleEMDR);

        // Add timer select dropdown
        timerSelect = document.createElement('select');
        timerSelect.style.position = 'absolute';
        timerSelect.style.top = '20px';
        timerSelect.style.left = '150px';
        timerSelect.style.fontSize = '16px';
        timerSelect.style.padding = '5px';
        timerSelect.style.borderRadius = '5px';
        timerSelect.style.border = '1px solid #ccc';

        // Populate dropdown options from 5 to 45 minutes in 5-minute increments
        for (let i = 5; i <= 45; i += 5) {
            let option = document.createElement('option');
            option.value = i;
            option.textContent = i + ' min';
            timerSelect.appendChild(option);
        }
        timerSelect.value = '15'; // Default to 15 minutes
        document.body.appendChild(timerSelect);

        // Add timer display
        timerDisplay = document.createElement('span');
        timerDisplay.style.position = 'absolute';
        timerDisplay.style.top = '20px';
        timerDisplay.style.left = '250px';
        timerDisplay.style.fontSize = '18px';
        timerDisplay.style.color = 'white';
        document.body.appendChild(timerDisplay);

        // Add event listeners for button visibility
        document.addEventListener('mousemove', showButton);
        startButton.addEventListener('mouseout', hideButtonIfRunning);
    }

    function toggleEMDR() {
        if (!isRunning) {
            isRunning = true;
            startButton.textContent = 'Stop EMDR';

            // Get the timer duration from dropdown
            timerDuration = parseInt(timerSelect.value) * 60; // Convert minutes to seconds
            timeRemaining = timerDuration;
            if (timeRemaining > 0) {
                // Start the timer
                timerInterval = setInterval(updateTimer, 1000); // Update every second
                updateTimerDisplay();
            } else {
                // If no timer is set, clear the display
                timerDisplay.textContent = '';
            }

            requestAnimationFrame(draw);
            hideButtonIfRunning();
        } else {
            stopEMDR();
        }
    }

    function stopEMDR() {
        isRunning = false;
        startButton.textContent = 'Start EMDR';
        drawStaticState();
        showButton();

        // Clear the timer if it's running
        if (timerInterval) {
            clearInterval(timerInterval);
            timerInterval = null;
        }

        // Clear the timer display
        timerDisplay.textContent = '';
    }

    function updateTimer() {
        if (timeRemaining > 0) {
            timeRemaining--;
            updateTimerDisplay();
            if (timeRemaining <= 0) {
                // Time's up, stop EMDR
                stopEMDR();
            }
        }
    }

    function updateTimerDisplay() {
        let minutes = Math.floor(timeRemaining / 60);
        let seconds = timeRemaining % 60;
        timerDisplay.textContent = minutes + ':' + seconds.toString().padStart(2, '0');
    }

    function showButton() {
        startButton.style.opacity = '1';
        startButton.style.pointerEvents = 'auto';
    }

    function hideButtonIfRunning() {
        if (isRunning) {
            startButton.style.opacity = '0';
            startButton.style.pointerEvents = 'none';
        }
    }

    function resizeCanvas() {
        const canvas = document.getElementById('emdrCanvas');
        canvasWidth = canvas.width = window.innerWidth;
        canvasHeight = canvas.height = window.innerHeight;
        dotY = canvasHeight / 2;
        if (!isRunning) {
            drawStaticState();
        }
    }

    function handleKeyPress(event) {
        if (event.key === 'ArrowUp') {
            dotSpeed += 1;
        } else if (event.key === 'ArrowDown') {
            dotSpeed = Math.max(1, dotSpeed - 1);
        }
    }

    function drawStaticState() {
        const canvas = document.getElementById('emdrCanvas');
        const ctx = canvas.getContext('2d');

        ctx.fillStyle = '#3B5998';  // Dark blue background
        ctx.fillRect(0, 0, canvasWidth, canvasHeight);

        ctx.fillStyle = '#FF0000';  // Red dot
        ctx.beginPath();
        ctx.arc(canvasWidth / 2, canvasHeight / 2, dotRadius, 0, Math.PI * 2);
        ctx.fill();
    }

    function draw() {
        if (!isRunning) return;

        const canvas = document.getElementById('emdrCanvas');
        const ctx = canvas.getContext('2d');

        ctx.fillStyle = '#3B5998';  // Dark blue background
        ctx.fillRect(0, 0, canvasWidth, canvasHeight);

        dotX += dotSpeed * direction;

        const edgeThreshold = 20;
        if (direction === 1 && dotX >= canvasWidth - dotRadius - edgeThreshold) {
            clickRightSound.play().catch(e => console.error("Error playing sound:", e));
        } else if (direction === -1 && dotX <= dotRadius + edgeThreshold) {
            clickLeftSound.play().catch(e => console.error("Error playing sound:", e));
        }

        if (dotX <= dotRadius || dotX >= canvasWidth - dotRadius) {
            direction *= -1;
        }

        ctx.fillStyle = '#FF0000';  // Red dot
        ctx.beginPath();
        ctx.arc(dotX, dotY, dotRadius, 0, Math.PI * 2);
        ctx.fill();

        requestAnimationFrame(draw);
    }

    document.addEventListener('DOMContentLoaded', setup);
    """

    html_content = Html(
        Head(
            Title("EMDR Tool"),
            Script(js_code),
            Style("""
                body { margin: 0; overflow: hidden; }
                canvas { display: block; }
                button { 
                    font-size: 16px; 
                    padding: 10px 20px; 
                    cursor: pointer;
                    background-color: #4CAF50;
                    color: white;
                    border: none;
                    border-radius: 5px;
                    transition: opacity 0.3s ease-in-out, background-color 0.3s ease;
                }
                button:hover {
                    background-color: #45a049;
                }
                select {
                    font-size: 16px;
                    padding: 5px;
                    border-radius: 5px;
                    border: 1px solid #ccc;
                }
                span {
                    font-size: 18px;
                    color: white;
                }
            """)
        ),
        Body(
            Canvas(id="emdrCanvas"),
        )
    )

    return html_content


@rt('/static/{filename}')
def get(filename: str):
    file_path = os.path.join('resources', filename)
    if not os.path.exists(file_path):
        raise NotFoundError(f"File {filename} not found")
    return FileResponse(file_path)


@rt("/{fname:path}.{ext:static}")
async def static(fname: str, ext: str):
    return FileResponse(f"static/{fname}.{ext}")


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("app", host="0.0.0.0", port=5001, reload=True)
def update_existing_tasks():
    # Fetch tasks that don't have category or urgency assigned
    cursor.execute("SELECT id, description FROM tasks WHERE status != 'completed'")
    tasks_to_update = cursor.fetchall()

    print(f"Found {len(tasks_to_update)} tasks to update.")

    for task_id, description in tasks_to_update:
        print(f"Updating task {task_id}: {description}")
        category, urgency = categorize_task(description)
        print(f"Assigned category: {category}, urgency: {urgency}")

        cursor.execute(
            'UPDATE tasks SET category = ?, urgency = ? WHERE id = ?',
            (category, urgency, task_id)
        )

    conn.commit()
    print(f"Updated {len(tasks_to_update)} tasks with category and urgency.")

# update_existing_tasks()
serve(port=5007)
