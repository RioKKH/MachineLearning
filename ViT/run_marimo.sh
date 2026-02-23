#!/bin/bash

uv run marimo edit --host 0.0.0.0 --port 2718 $1 &

return 0
