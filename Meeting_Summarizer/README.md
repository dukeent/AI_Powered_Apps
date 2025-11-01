# Meeting Transcript Summarizer

This application automatically processes meeting transcripts and generates concise summaries using Azure OpenAI's GPT models.

## Problem Statement

In today's fast-paced work environment, attending every meeting or reviewing lengthy meeting notes can be challenging. An AI-powered tool that automatically generates concise summaries from meeting transcripts is invaluable.

This application aims to process raw meeting transcripts and produce clear, concise summaries highlighting key points, decisions, and action items.

## Features

1. **Automated Processing**
   - Automatically processes all meeting transcripts in the `transcripts` directory
   - No manual input required
   - Generates structured summaries with key points, decisions, and action items

2. **Comprehensive Logging**
   - Saves both original transcripts and summaries
   - Timestamps all outputs
   - Maintains detailed processing logs

## Setup

1. Create a `.env` file with your Azure OpenAI credentials:
```
AZURE_OPENAI_ENDPOINT="your-endpoint"
AZURE_OPENAI_API_KEY="your-api-key"
AZURE_DEPLOYMENT_NAME="GPT-4o-mini"
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python main.py
```

## Sample Transcripts

The application includes 5 sample meeting transcripts:

1. `meeting1.txt` - Product Team Standup
2. `meeting2.txt` - Client Review Meeting
3. `meeting3.txt` - Architecture Review
4. `meeting4.txt` - Budget Planning 2026
5. `meeting5.txt` - Team Training Planning

Each transcript represents different types of meetings to demonstrate the application's versatility in handling various meeting contexts.

## Output Format

The application generates a log file with the following structure for each transcript:

```
File: [transcript_path]
--------------------------------------------------
Original Transcript:
[Original meeting transcript content]

Summary:
Key Points:
- [Key point 1]
- [Key point 2]

Decisions:
- [Decision 1]
- [Decision 2]

Action Items:
- [Action item 1]
- [Action item 2]
==================================================
```

## Directory Structure

```
Assignment_03/
├── main.py              # Main application code
├── requirements.txt     # Python dependencies
├── .env                # Environment variables (not in repo)
├── README.md           # This documentation
└── transcripts/        # Sample meeting transcripts
    ├── meeting1.txt
    ├── meeting2.txt
    ├── meeting3.txt
    ├── meeting4.txt
    └── meeting5.txt
```