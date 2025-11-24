#!/bin/bash
# Script to check logs after testing upload with biomdlore accounts

echo "=== Recent Upload Attempts ==="
echo ""
tail -100 /opt/mcads/app/logs/gunicorn_error.log | grep -A 3 "Upload attempt\|Form validation\|Form errors\|Returning success"
echo ""
echo "=== Any Errors or Exceptions ==="
echo ""
tail -100 /opt/mcads/app/logs/gunicorn_error.log | grep -i "error\|exception\|traceback" | tail -20

