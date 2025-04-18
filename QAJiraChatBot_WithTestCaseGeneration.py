import streamlit as st
import pandas as pd
import requests
from requests.auth import HTTPBasicAuth
from urllib.parse import urljoin
import re
import time
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.tools import tool

# --- JIRA CONFIGURATION ---
JIRA_CONFIG = {
    'JIRA_URL': "http://localhost:8080",
    'JIRA_USERNAME': "jiratestuser4",
    'JIRA_PASSWORD': "sriram10",
    'TEST_ISSUE_TYPE': "Test"
}

# --- JIRA Client Setup ---
try:
    from jira import JIRA, JIRAError

    jira_options = {'server': JIRA_CONFIG['JIRA_URL']}
    jira = JIRA(options=jira_options, basic_auth=(JIRA_CONFIG['JIRA_USERNAME'], JIRA_CONFIG['JIRA_PASSWORD']))
except ImportError:
    jira = None
    st.warning("JIRA Python library not installed. Some chatbot features may be limited.")


# --- LLM FUNCTIONS ---
def call_llm(prompt):
    if llm_choice == "OpenAI":
        from openai import OpenAI
        client = OpenAI(api_key=openai_key)
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a QA test expert."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    elif llm_choice == "Ollama":
        response = requests.post(
            f"{ollama_endpoint}/api/chat",
            json={
                "model": ollama_model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False
            }
        )
        return response.json()["message"]["content"].strip()
    elif llm_choice == "Groq":
        headers = {"Authorization": f"Bearer {groq_key}"}
        json_data = {
            "model": groq_model,
            "messages": [{"role": "user", "content": prompt}]
        }
        response = requests.post(groq_endpoint, headers=headers, json=json_data)
        return response.json()["choices"][0]["message"]["content"].strip()


# --- JIRA TOOLS for Chatbot ---
@tool
def get_linked_tests(issue_key: str) -> str:
    """Retrieves all test cases linked to a Jira user story or issue.

    Args:
        issue_key: The key of the Jira issue (e.g., 'PROJ-123')

    Returns:
        String containing formatted list of linked test cases with their statuses
    """
    try:
        # Get the issue
        issue = jira.issue(issue_key)

        # Find all test case links (assuming 'Tests' link type)
        test_cases = []
        for link in issue.fields.issuelinks:
            if hasattr(link, 'inwardIssue') and link.type.name == 'Tests':
                test_case = link.inwardIssue
                test_cases.append({
                    'key': test_case.key,
                    'summary': test_case.fields.summary,
                    'status': test_case.fields.status.name
                })
            elif hasattr(link, 'outwardIssue') and link.type.name == 'Tests':
                test_case = link.outwardIssue
                test_cases.append({
                    'key': test_case.key,
                    'summary': test_case.fields.summary,
                    'status': test_case.fields.status.name
                })

        if not test_cases:
            return f"No test cases linked to {issue_key}"

        # Format the response
        response = f"📋 Test Cases Linked to {issue_key}:\n"
        for test in test_cases:
            response += f"- {test['key']}: {test['summary']} (Status: {test['status']})\n"

        return response

    except JIRAError as e:
        return f"❌ Error retrieving linked tests: {str(e)}"

@tool
def close_issue(issue_id: str) -> str:
    """Closes a Jira issue given an issue ID, handling errors gracefully."""
    try:
        issue = jira.issue(issue_id)
        transitions = jira.transitions(issue)

        transition_names = [t['name'].lower() for t in transitions]
        print(f"Available transitions for {issue_id}: {transition_names}")

        for transition in transitions:
            if transition['name'].lower() in ["done", "closed"]:
                jira.transition_issue(issue, transition['id'])
                return f"✅ Issue {issue_id} has been closed."

        return f"❌ Could not close issue {issue_id}. Valid transitions: {transition_names}"

    except JIRAError as e:
        return f"❌ Error: {str(e)}"


@tool
def create_issue(project_key: str, summary: str, description: str, issue_type: str = "Task") -> str:
    """Creates a Jira issue given project key, summary, and description."""
    try:
        issue_dict = {
            "project": {"key": project_key},
            "summary": summary,
            "description": description,
            "issuetype": {"name": issue_type},
        }
        new_issue = jira.create_issue(fields=issue_dict)
        return f"🎉 Issue {new_issue.key} created successfully!"

    except JIRAError as e:
        return f"❌ Error: {str(e)}"


@tool
def update_issue_status(issue_id: str, status: str) -> str:
    """Updates a Jira issue status given issue ID and new status."""
    try:
        issue = jira.issue(issue_id)
        transitions = jira.transitions(issue)

        transition_mapping = {t['name'].lower(): t['id'] for t in transitions}
        print(f"Available transitions for {issue_id}: {transition_mapping}")  # Debugging

        if status.lower() in transition_mapping:
            jira.transition_issue(issue, transition_mapping[status.lower()])
            return f"🔄 Issue {issue_id} updated to '{status}'."
        else:
            return f"❌ No valid transition for '{status}'. Available: {list(transition_mapping.keys())}"

    except JIRAError as e:
        return f"❌ Error: {str(e)}"


@tool
def get_issue_status(issue_id: str) -> str:
    """Retrieves the current status of a Jira issue."""
    try:
        issue = jira.issue(issue_id)
        return f"📌 Issue {issue_id} is currently in status: {issue.fields.status.name}"

    except JIRAError as e:
        return f"❌ Error: {str(e)}"


@tool
def summarize_issue(issue_id: str) -> str:
    """Fetches and summarizes a Jira issue given its ID."""
    try:
        issue = jira.issue(issue_id)

        # Extract Key Details
        summary = issue.fields.summary
        description = issue.fields.description or "No description available."
        status = issue.fields.status.name
        comments = [comment.body for comment in getattr(issue.fields.comment, 'comments', [])[-3:]]

        issue_details = f"""
        **Issue ID:** {issue_id}
        **Summary:** {summary}
        **Status:** {status}
        **Description:** {description}
        **Recent Comments:** {comments or 'No recent comments'}
        """

        prompt = f"Summarize the following Jira issue:\n\n{issue_details}"
        response = call_llm(prompt)

        return f"📌 **Issue Summary for {issue_id}:**\n{response}"

    except JIRAError as e:
        return f"❌ Error: {str(e)}"


@tool
def get_backlog_issues(project_key: str, max_results: int = 5) -> str:
    """Retrieves backlog issues for a specific project."""
    try:
        jql_query = f"project = '{project_key}' AND sprint is EMPTY AND statusCategory != Done ORDER BY created DESC"
        issues = jira.search_issues(jql_query, maxResults=max_results)

        if not issues:
            return f"📂 No backlog issues found in {project_key}."

        issue_list = "\n".join([f"- {issue.key}: {issue.fields.summary}" for issue in issues])
        return f"📂 Backlog Issues in {project_key}:\n{issue_list}"
    except JIRAError as e:
        return f"❌ Error: {str(e)}"


@tool
def get_active_sprint(project_key: str) -> str:
    """Retrieves the active sprint for a given Jira project."""
    try:
        boards = jira.boards(name=project_key)
        if not boards:
            return f"❌ No boards found for project {project_key}."

        board_id = boards[0].id
        sprints = jira.sprints(board_id)

        active_sprint = next((sprint for sprint in sprints if sprint.state == "active"), None)

        if not active_sprint:
            return f"⏳ No active sprint found for project {project_key}."

        return f"🚀 Active Sprint in {project_key}: {active_sprint.name} (ID: {active_sprint.id})"

    except JIRAError as e:
        return f"❌ Error: {str(e)}"


def initialize_jira_agent():
    if 'jira_agent' not in st.session_state:
        llm = None

        if llm_choice == "OpenAI":
            llm = ChatOpenAI(model="gpt-4", openai_api_key=openai_key, temperature=0)
        elif llm_choice == "Ollama":
            from langchain_community.llms import Ollama
            llm = Ollama(base_url=ollama_endpoint, model=ollama_model)
        elif llm_choice == "Groq":
            from langchain_groq import ChatGroq
            llm = ChatGroq(temperature=0, groq_api_key=groq_key, model_name=groq_model)

        if not llm:
            st.error(f"Could not initialize {llm_choice} LLM")
            return None

        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        tools = [
            close_issue,
            create_issue,
            update_issue_status,
            get_issue_status,
            summarize_issue,
            get_backlog_issues,
            get_active_sprint,
            get_linked_tests
        ]

        st.session_state.jira_agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.OPENAI_FUNCTIONS,
            memory=memory,
            verbose=True
        )
    return st.session_state.jira_agent


# --- JIRA FUNCTIONS for Test Case Generator ---
def get_jira_issue_details(issue_key):
    url = urljoin(JIRA_CONFIG['JIRA_URL'], f"/rest/api/2/issue/{issue_key}")
    auth = HTTPBasicAuth(JIRA_CONFIG['JIRA_USERNAME'], JIRA_CONFIG['JIRA_PASSWORD'])

    try:
        response = requests.get(url, auth=auth)
        response.raise_for_status()
        issue_data = response.json()

        return {
            'key': issue_data['key'],
            'summary': issue_data['fields']['summary'],
            'description': issue_data['fields'].get('description', ''),
            'project_key': issue_data['fields']['project']['key'],
            'status': issue_data['fields']['status']['name']
        }
    except Exception as e:
        st.error(f"Error fetching Jira issue: {str(e)}")
        return None


def create_jira_test_case(test_data, user_story):
    url = urljoin(JIRA_CONFIG['JIRA_URL'], "/rest/api/2/issue")
    auth = HTTPBasicAuth(JIRA_CONFIG['JIRA_USERNAME'], JIRA_CONFIG['JIRA_PASSWORD'])
    headers = {"Content-Type": "application/json"}

    steps = [step.strip() for step in test_data['Steps'].split('\n') if step.strip()]

    test_issue = {
        "fields": {
            "project": {"key": user_story['project_key']},
            "issuetype": {"name": JIRA_CONFIG['TEST_ISSUE_TYPE']},
            "summary": test_data['Title'],
            "description": f"Steps:\n{test_data['Steps']}\n\nExpected Result:\n{test_data['Expected Result']}",
            "customfield_10115": {
                "steps": [{
                    "index": i + 1,
                    "fields": {
                        "Action": step,
                        "Data": "",
                        "Expected Result": test_data['Expected Result']
                    }
                } for i, step in enumerate(steps)]
            }
        }
    }

    try:
        response = requests.post(url, auth=auth, headers=headers, json=test_issue)
        response.raise_for_status()
        test_key = response.json().get('key')

        # Link to user story
        link_data = {
            "type": {"name": "Tests"},
            "inwardIssue": {"key": test_key},
            "outwardIssue": {"key": user_story['key']}
        }
        requests.post(
            urljoin(JIRA_CONFIG['JIRA_URL'], "/rest/api/2/issueLink"),
            auth=auth,
            headers=headers,
            json=link_data
        )
        return test_key
    except Exception as e:
        st.error(f"Error creating test: {str(e)}")
        return None


# --- PARSER ---
def parse_llm_response(response):
    titles = re.findall(r"(?:Title|Test Case):\s*(.*)", response)
    steps_raw = re.findall(r"Steps:\n((?:- .+\n?)+)", response)
    expected = re.findall(r"Expected Result:\s*(.*)", response)

    rows = []
    for i in range(len(titles)):
        title = titles[i].strip()
        steps_list = re.findall(r"- (.+)", steps_raw[i]) if i < len(steps_raw) else []
        steps = "\n".join(f"{j + 1}. {step}" for j, step in enumerate(steps_list))
        result = expected[i].strip() if i < len(expected) else ""
        rows.append({
            "Include": True,
            "Title": title,
            "Steps": steps,
            "Expected Result": result,
        })
    return pd.DataFrame(rows)


# --- STREAMLIT APP ---
st.set_page_config(page_title="Test Case Generator", layout="wide")
st.title("🧪 AI Test Case Generator")

# Sidebar - LLM Configuration
with st.sidebar:
    st.header("🔧 LLM Settings")
    llm_choice = st.selectbox("Choose LLM", ["OpenAI", "Ollama", "Groq"])

    if llm_choice == "OpenAI":
        openai_key = st.text_input("OpenAI API Key", type="password")
    elif llm_choice == "Ollama":
        ollama_endpoint = st.text_input("Ollama Endpoint", value="http://localhost:11434")
        ollama_model = st.text_input("Ollama Model", value="llama2")
    elif llm_choice == "Groq":
        groq_endpoint = st.text_input("Groq Endpoint", value="https://api.groq.com/openai/v1/chat/completions")
        groq_key = st.text_input("Groq API Key", type="password")
        groq_model = st.text_input("Groq Model", value="mixtral-8x7b-32768")

# Main App - Tabs
tab1, tab2 = st.tabs(["Test Case Generator", "JIRA Chatbot"])

with tab1:
    issue_key = st.text_input("🔍 Enter Jira User Story Key (e.g., AIP-123):", "").strip().upper()

    if issue_key:
        with st.spinner("Fetching user story details..."):
            user_story = get_jira_issue_details(issue_key)

            if user_story:
                st.success(f"Fetched: {user_story['key']} - {user_story['summary']} (Status: {user_story['status']})")

                # Generate Test Cases Button
                if st.button("💡 Generate Test Cases"):
                    with st.spinner("Analyzing user story..."):
                        prompt = f"""
                        As a QA expert, suggest detailed test cases for:

                        User Story: {user_story['summary']}
                        Description: {user_story['description']}

                        Include positive, negative and edge cases with steps for each case. Format each as:
                        Title: <title>
                        Steps:
                        - step 1
                        - step 2
                        Expected Result: <result>
                        """

                        try:
                            response = call_llm(prompt)
                            st.session_state.suggested_tests = parse_llm_response(response)
                        except Exception as e:
                            st.error(f"LLM Error: {str(e)}")

    # Display and Edit Suggested Tests
    if 'suggested_tests' in st.session_state and not st.session_state.suggested_tests.empty:
        st.markdown("### 🧪 Suggested Test Cases")

        # Editable table with checkboxes
        edited_df = st.data_editor(
            st.session_state.suggested_tests,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Include": st.column_config.CheckboxColumn("Create", default=True),
                "Title": "Test Case",
                "Steps": "Test Steps",
                "Expected Result": "Expected Result"
            },
            num_rows="dynamic"
        )

        # Add new test row
        if st.button("➕ Add Custom Test Case"):
            new_row = pd.DataFrame([{
                "Include": True,
                "Title": "",
                "Steps": "",
                "Expected Result": ""
            }])
            st.session_state.suggested_tests = pd.concat([st.session_state.suggested_tests, new_row])
            st.rerun()

        # Push to Jira
        if st.button("🚀 Push Selected to Jira", type="primary"):
            selected_tests = edited_df[edited_df["Include"] == True]

            if selected_tests.empty:
                st.warning("No test cases selected!")
            else:
                progress_bar = st.progress(0)
                created_tests = []

                for i, (_, test_case) in enumerate(selected_tests.iterrows()):
                    test_key = create_jira_test_case(test_case, user_story)
                    if test_key:
                        created_tests.append(test_key)
                    progress_bar.progress((i + 1) / len(selected_tests))

                if created_tests:
                    st.success(f"✅ Created {len(created_tests)} test cases!")
                    for key in created_tests:
                        st.markdown(f"- {key}: {urljoin(JIRA_CONFIG['JIRA_URL'], f'/browse/{key}')}")
                    st.balloons()
                else:
                    st.error("❌ Failed to create test cases")

    with tab2:
        st.markdown("## 🤖 JIRA Assistant")
        st.markdown("Chat with your JIRA assistant to manage issues, sprints, and more!")

        # Initialize chat history
        if "jira_chat_history" not in st.session_state:
            st.session_state.jira_chat_history = []

        # Display chat history
        for message in st.session_state.jira_chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("Ask about JIRA issues (e.g., 'What's the status of ABC-123?')"):
            # Add user message to chat history
            st.session_state.jira_chat_history.append({"role": "user", "content": prompt})

            with st.chat_message("user"):
                st.markdown(prompt)

            # Process command
            if llm_choice != "OpenAI":
                response = "⚠️ JIRA Chatbot requires OpenAI LLM. Please select OpenAI in the sidebar."
            else:
                agent = initialize_jira_agent()
                if agent:
                    with st.spinner("Thinking..."):
                        try:
                            response = agent.run(prompt)
                        except Exception as e:
                            response = f"❌ Error processing your request: {str(e)}"
                else:
                    response = "⚠️ Could not initialize JIRA agent. Check your OpenAI API key."

            # Add assistant response to chat history
            st.session_state.jira_chat_history.append({"role": "assistant", "content": response})

            with st.chat_message("assistant"):
                st.markdown(response)

        st.markdown("""
        ### Example Commands:
        - "What's the status of ABC-123?"
        - "Close issue ABC-123"
        - "Create a new task in project XYZ with title 'Fix bug'"
        - "What's in the backlog for project ABC?"
        - "Summarize issue DEF-456"
        """)

        if st.button("Clear Chat History", key="clear_chat"):
            st.session_state.jira_chat_history = []
            st.rerun()