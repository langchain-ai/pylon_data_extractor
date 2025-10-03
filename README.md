# Pylon Data Extractor

Extended data replication utility for Pylon using the Pylon REST API. This tool extracts data from Pylon and replicates it to Google BigQuery for analytics and reporting purposes.

## Features

- **Multiple Object Types**: Extract accounts, issues, messages, contacts, users, and teams
- **Incremental Replication**: Support for delta replication using timestamp-based filtering
- **Batch Processing**: Configurable batch sizes for optimal performance
- **Rate Limiting**: Built-in rate limiting and retry logic for API stability
- **BigQuery Integration**: Direct integration with Google BigQuery for data warehousing
- **Comprehensive Logging**: Structured logging with configurable levels
- **Configuration Management**: Flexible configuration via environment variables or config files

## Requirements

- Python 3.9+
- Google Cloud BigQuery access
- Pylon API access credentials

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd pylon_data_extractor
   ```

2. **Install dependencies**:
   ```bash
   uv sync
   ```

3. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```
- Update the keys and service account JSON paths in the env file

## Usage

### Command Line Interface

The tool provides three main commands: `replicate`, `truncate`, `classify`, and `close`

#### Replicate Data

Extract and replicate data from Pylon to BigQuery:

```bash
# Replicate all accounts
uv run pylon-extract replicate accounts

# Replicate with custom batch size
uv run pylon-extract replicate accounts --batch-size 500

# Replicate all object types
uv run pylon-extract replicate all

# Save data after each API page (useful for large datasets)
uv run pylon-extract replicate accounts --save-each-page

# Limit the number of records extracted
uv run pylon-extract replicate issues --max-records 1000

# Message replication methods
uv run pylon-extract replicate messages --created-start "2024-09-03" --created-end "2024-12-31"
uv run pylon-extract replicate messages --issue-id "XYZ"
uv run pylon-extract replicate messages --states "on_hold"

```

#### Truncate Tables

Clear BigQuery tables before fresh loads:

```bash
# Truncate a specific table
uv run pylon-extract truncate int__pylon_accounts
```

#### Classify Issues

Automatically classify closed Pylon issues with missing resolution or category fields using AI:

```bash
# Classify both resolution and category for all applicable issues
uv run pylon-extract classify

# Classify only resolution field
uv run pylon-extract classify --fields resolution

# Classify only category field
uv run pylon-extract classify --fields category

# Classify with custom batch size and max records
uv run pylon-extract classify --batch-size 50 --max-records 100

# Classify a single specific issue
uv run pylon-extract classify --issue-id "d5a04df6-6882-454b-bc8f-0d91fb688762"

# Classification with debug logging
uv run pylon-extract classify --log-level DEBUG
```

The classify command:
- Identifies closed issues with missing resolution or category values
- Retrieves conversation history for each issue from BigQuery
- Uses OpenAI GPT-4 to classify issues based on conversation content
- Only updates issues when confidence score is above 0.8
- Automatically tags updated issues with "auto-classified"
- Supports both individual and combined field classification

#### Close Tickets

Automatically close eligible Slack tickets based on question analysis:

```bash
# Close eligible tickets with default settings
uv run pylon-extract close

# Close with custom batch size and max records
uv run pylon-extract close --batch-size 20 --max-records 50

# Process a single specific issue
uv run pylon-extract close --issue-id "d5a04df6-6882-454b-bc8f-0d91fb688762"

# Close with debug logging
uv run pylon-extract close --log-level DEBUG
```

The close command:
- Identifies "on customer" Slack issues with resolution and category set
- Filters for issues older than 1 week since last message
- Analyzes conversation history to extract customer questions
- Determines if all customer questions have been resolved
- Automatically closes issues where all questions are answered
- Only processes Slack tickets (email tickets are excluded)

#### Available Object Types

- `accounts` - Account/organization data
- `issues` - Support tickets and issues
- `messages` - Messages within issues
- `contacts` - Contact information
- `users` - User accounts
- `teams` - Team information
- `all` - All supported object types

### Configuration Files

You can also use YAML configuration files:

```yaml
# config/production.yaml
pylon:
  api_key: "your_api_key"
  base_url: "https://api.pylon.com"
  batch_size: 100

bigquery:
  project_id: "your_project"
  dataset: "pylon_data"
  location: "US"

replication:
  max_retries: 3
  rate_limit_calls: 100
  rate_limit_period: 60
```

Use with:
```bash
uv run pylon-extract replicate accounts
```

## Development

### Setup Development Environment

```bash
# Install with development dependencies
uv sync --dev

# Run tests
uv run pytest
```

### Project Structure

```
pylon_data_extractor/
├── src/
│   ├── __init__.py
│   ├── main.py              # CLI entry point
│   ├── config.py            # Configuration management
│   ├── pylon_client.py      # Pylon API client
│   ├── replicator.py        # Data replication logic
│   └── bigquery_utils.py    # BigQuery utilities
├── tests/                   # Test suite
├── config/                  # Configuration files
├── pyproject.toml          # Project dependencies and metadata
└── README.md               # This file
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_pylon_client.py

# Run with verbose output
uv run pytest -v

# Run tests with coverage report
uv run pytest --cov=src --cov-report=html
```

## Data Schema

The tool creates the following BigQuery tables:

- `int__pylon_accounts` - Account data
- `int__pylon_issues` - Issue/ticket data
- `int__pylon_messages` - Message data within issues
- `int__pylon_contacts` - Contact information
- `int__pylon_users` - User account data
- `int__pylon_teams` - Team data

Each table includes metadata columns:
- `_extracted_at` - Timestamp when data was extracted
- `_pylon_id` - Original Pylon object ID

## Error Handling

The tool includes comprehensive error handling:

- **API Rate Limits**: Automatic retry with exponential backoff
- **Network Errors**: Configurable retry attempts
- **Data Validation**: Schema validation before BigQuery insertion
- **Partial Failures**: Continue processing other objects if one fails

## Logging

Structured logging is provided with configurable levels:

```bash
# Debug level logging
uv run pylon-extract replicate accounts --log-level DEBUG

# Available levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
```

## Performance Considerations

- **Batch Size**: Larger batches reduce API calls but increase memory usage
- **Rate Limiting**: Respect Pylon's API rate limits to avoid throttling
- **BigQuery Streaming**: Data is streamed to BigQuery for better performance
