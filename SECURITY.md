# Security

MemCoder executes model-generated Python. Treat all generated code and externally supplied tests as
untrusted.

The local executor provides disposable directories, a stripped environment, isolated Python mode,
timeouts, output limits, and Unix process resource limits. It does not reliably block network access,
host filesystem reads, kernel attacks, or vulnerabilities in the Python interpreter and installed packages.

For untrusted workloads:

1. Run the entire application in a disposable container or virtual machine.
2. Do not mount personal directories, SSH agents, cloud credentials, or Docker sockets.
3. Disable networking at the container/runtime layer.
4. Apply CPU, memory, process, and filesystem quotas outside Python.
5. Destroy the environment after the run.

Never place API keys in benchmark tasks, tests, generated files, or committed `.env` files.

