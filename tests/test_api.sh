#!/bin/bash

# Simple test script for the BPF Profiler API

echo "Testing BPF Profiler API..."

# Test 1: Check server status
echo "1. Testing server status..."
curl -s http://localhost:8080/api/status | jq .
echo

# Test 2: Start a profile session (using current shell PID for testing)
CURRENT_PID=$$
echo "2. Starting profile session for PID $CURRENT_PID..."
SESSION_RESPONSE=$(curl -s -X POST http://localhost:8080/api/v1/analyzers/profile/start \
  -H "Content-Type: application/json" \
  -d "{
    \"duration\": 5,
    \"targets\": {
      \"pids\": [$CURRENT_PID]
    },
    \"config\": {
      \"frequency\": 99
    }
  }")

echo "$SESSION_RESPONSE" | jq .
SESSION_ID=$(echo "$SESSION_RESPONSE" | jq -r '.session_id // empty')

if [ -z "$SESSION_ID" ]; then
    echo "Failed to get session ID. Exiting."
    exit 1
fi

echo "Session ID: $SESSION_ID"
echo

# Test 3: Check session status
echo "3. Checking session status..."
curl -s http://localhost:8080/api/v1/analyzers/profile/$SESSION_ID/status | jq .
echo

# Test 4: Wait a bit then get flamegraph data
echo "4. Waiting 6 seconds for profiling to complete..."
sleep 6

echo "5. Getting flamegraph data..."
curl -s http://localhost:8080/api/v1/analyzers/profile/$SESSION_ID/views | jq '.views.flamegraph.data | {analyzer_name, success, total_samples}'
echo

echo "API test completed!"
echo "You can now open http://localhost:8080 in your browser to use the demo interface." 