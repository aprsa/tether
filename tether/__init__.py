"""tether -- Task Execution and Transfer Handler for External Resources.

A synchronous bridge to a remote compute resource, over a single reused SSH
connection.

    import tether

    with tether.server("terra") as terra:
        print(terra.info())
        for p in terra.partitions():
            print(p)
        for job in terra.queue(user=terra.whoami()):
            print(job)

Design axiom: the remote filesystem is the source of truth and the SSH
connection is a disposable, reconnectable link. Jobs outlive Python
sessions.
"""

from .config import Config, EnvironmentConfig, ServerConfig, ServerKind, EnvironmentKind, load_config
from .errors import (
    ConfigError,
    RemoteCommandError,
    SlurmError,
    TetherError,
    LinkError,
)
from .server import Host, Server, SlurmServer, server
from .slurm import Job, Partition, parse_duration
from .link import Result, Link

__version__ = "0.1.0"

__all__ = [
    'Config',
    'ConfigError',
    'EnvironmentConfig',
    'EnvironmentKind',
    'Host',
    'Job',
    'Partition',
    'RemoteCommandError',
    'Result',
    'Server',
    'ServerConfig',
    'ServerKind',
    'SlurmError',
    'SlurmServer',
    'TetherError',
    'Link',
    'LinkError',
    'load_config',
    'parse_duration',
    'server',
    '__version__',
]
