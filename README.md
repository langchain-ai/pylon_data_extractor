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

The tool provides two main commands: `replicate` and `truncate`.

#### Replicate Data

Extract and replicate data from Pylon to BigQuery:

```bash
# Replicate all accounts
uv run pylon-extract replicate accounts

# Replicate with incremental sync (since specific date)
uv run pylon-extract replicate accounts --updated-dt 2024-01-01

# Replicate with custom batch size
uv run pylon-extract replicate accounts --batch-size 500

# Replicate all object types
uv run pylon-extract replicate all

# Save data after each API page (useful for large datasets)
uv run pylon-extract replicate accounts --save-each-page

# Limit the number of records extracted
uv run pylon-extract replicate accounts --max-records 1000
```

#### Truncate Tables

Clear BigQuery tables before fresh loads:

```bash
# Truncate a specific table
uv run pylon-extract truncate int__pylon_accounts
```

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
uv run pylon-extract replicate accounts --config config/production.yaml
```

## Development

### Setup Development Environment

```bash
# Install with development dependencies
uv sync --dev

# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=src

# Format code
uv run black src/ tests/

# Lint code
uv run ruff check src/ tests/

# Type checking
uv run mypy src/
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
- **Incremental Sync**: Use `--updated-dt` for efficient delta replication
- **BigQuery Streaming**: Data is streamed to BigQuery for better performance
