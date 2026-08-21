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

from . import environment
from .config import (
    EnvironmentConfig,
    EnvironmentKind,
    ServerConfig,
    ServerKind,
    delete_server,
    load_server,
    save_server,
    server_path,
    list_servers,
)
from .errors import (
    EnvActivationError,
    ConfigError,
    RemoteCommandError,
    SlurmError,
    TetherError,
    LinkError,
)
from .server import EnvironmentInfo, Host, Server, SlurmServer, server
from .slurm import Job, Partition, parse_duration
from .link import Result, Link

__version__ = '0.1.0'

__all__ = [
    'EnvActivationError',
    'ConfigError',
    'EnvironmentConfig',
    'EnvironmentInfo',
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
    'environment',
    'delete_server',
    'load_server',
    'save_server',
    'server_path',
    'list_servers',
    'parse_duration',
    'server',
    '__version__',
]
