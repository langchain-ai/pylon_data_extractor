-- Intermediate Pylon Tables Creation Script
-- Project: langchain-prod
-- Dataset: src_pylon
-- Created: $(date)

-- Create intermediate table for Pylon issues
CREATE TABLE `langchain-prod.src_pylon.int__pylon_issues` (
  issue_id STRING NOT NULL,
  data JSON,
  updated_at TIMESTAMP
)
OPTIONS (
  description = "Intermediate table for Pylon issues data"
);

-- Create intermediate table for Pylon messages
CREATE TABLE `langchain-prod.src_pylon.int__pylon_messages` (
  issue_id STRING NOT NULL,
  message_id STRING NOT NULL,
  data JSON,
  updated_at TIMESTAMP
)
OPTIONS (
  description = "Intermediate table for Pylon messages data"
);

-- Create intermediate table for Pylon accounts
CREATE TABLE `langchain-prod.src_pylon.int__pylon_accounts` (
  account_id STRING NOT NULL,
  data JSON,
  updated_at TIMESTAMP
)
OPTIONS (
  description = "Intermediate table for Pylon accounts data"
);

-- Create intermediate table for Pylon contacts
CREATE TABLE `langchain-prod.src_pylon.int__pylon_contacts` (
  contact_id STRING NOT NULL,
  account_id STRING,
  data JSON,
  updated_at TIMESTAMP
)
OPTIONS (
  description = "Intermediate table for Pylon contacts data"
);

-- Create intermediate table for Pylon users
CREATE TABLE `langchain-prod.src_pylon.int__pylon_users` (
  user_id STRING NOT NULL,
  data JSON,
  updated_at TIMESTAMP
)
OPTIONS (
  description = "Intermediate table for Pylon users data"
);

-- Create intermediate table for Pylon teams
CREATE TABLE `langchain-prod.src_pylon.int__pylon_teams` (
  team_id STRING NOT NULL,
  data JSON,
  updated_at TIMESTAMP
)
OPTIONS (
  description = "Intermediate table for Pylon teams data"
);

-- Summary of created tables:
-- 1. int__pylon_issues - issue_id (STRING, NOT NULL), data (JSON), updated_at (TIMESTAMP)
-- 2. int__pylon_messages - issue_id (STRING, NOT NULL), message_id (STRING, NOT NULL), data (JSON), updated_at (TIMESTAMP)
-- 3. int__pylon_accounts - account_id (STRING, NOT NULL), data (JSON), updated_at (TIMESTAMP)
-- 4. int__pylon_contacts - contact_id (STRING, NOT NULL), account_id (STRING), data (JSON), updated_at (TIMESTAMP)
-- 5. int__pylon_users - user_id (STRING, NOT NULL), data (JSON), updated_at (TIMESTAMP)
-- 6. int__pylon_teams - team_id (STRING, NOT NULL), data (JSON), updated_at (TIMESTAMP)
