# Check the Docker sandbox on Windows

1. Install and start Docker Desktop. In PowerShell, run
   `docker info --format '{{.OSType}}'` and check that it prints `linux`.
   If it prints `windows`, switch Docker Desktop to Linux containers.
2. Run `desman sandbox setup` to see the download size, then run
   `desman sandbox setup --yes` to download the image. Run
   `desman sandbox status --json` and check for `"backend": "docker"`,
   `"available": true` and `"image_present": true`.
3. Enable `shell_execution` in `desman config` or with
   `desman permissions --grant shell_execution`. In the chat, ask DESMAN
   to write `answer.py` with a function returning `21 * 2`, write a pytest
   test that imports it and asserts it returns 42, then run
   `python3 -m pytest -q`. Check that the tool output says `1 passed`
   and that both files appear in the task's workspace.
4. Ask DESMAN to run an additional pytest test that attempts a socket
   connection to `1.1.1.1:53` with a two-second socket timeout. Expect an
   `OSError` and check that `socket.if_nameindex()` reports only `lo`.
5. In a separate task, ask DESMAN to run
   `python3 -c "import time; time.sleep(30)"` with a five-second
   `run_command` timeout. Check that the tool reports a timeout. Then run
   `docker ps -a --filter label=desman.sandbox=1`. With no other sandbox
   tests or tasks running, expect no containers in the list.
