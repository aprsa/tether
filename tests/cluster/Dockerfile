# A real, small Slurm cluster for tether's live tests.
# 
# This image emulates a single computing resource, so slurmctld, slurmd,
# slurmdbd, MariaDB and sshd all live together.

ARG UBUNTU_VERSION=26.04

FROM ubuntu:${UBUNTU_VERSION}

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        slurm-wlm slurmdbd munge \
        mariadb-server \
        openssh-server \
        environment-modules \
        python3 python3-venv \
        bzip2 ca-certificates curl procps \
    && rm -rf /var/lib/apt/lists/*

# The unprivileged account the tests log in as, and that jobs run under.
# `useradd` leaves `!` in the shadow password field, which sshd reads as a
# locked account and refuses even for key auth. `*` means "no password login"
# without the lock, which is what is actually wanted here.
RUN useradd --create-home --shell /bin/bash tether && usermod --password '*' tether

# slurm config:
COPY slurm.conf cgroup.conf /etc/slurm/
COPY slurmdbd.conf /etc/slurm/slurmdbd.conf
RUN chown slurm:slurm /etc/slurm/slurmdbd.conf \
    && chmod 0600 /etc/slurm/slurmdbd.conf

# database config:
RUN mkdir -p /run/mysqld && chown mysql:mysql /run/mysqld && mariadb-install-db --user=mysql --datadir=/var/lib/mysql > /dev/null

# create a test venv:
RUN python3 -m venv /opt/venvs/example

# install conda and create a test environment:
ARG MINIFORGE_VERSION=26.3.2-3
RUN base="https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}" \
    && name="Miniforge3-${MINIFORGE_VERSION}-Linux-$(uname -m).sh" \
    && cd /tmp \
    # --retry: GitHub's asset CDN answers the odd request with a 503.
    && curl -fsSL --retry 5 --retry-delay 2 --retry-all-errors -O "${base}/${name}" \
    && curl -fsSL --retry 5 --retry-delay 2 --retry-all-errors -O "${base}/${name}.sha256" \
    && sha256sum -c "${name}.sha256" \
    && bash "${name}" -b -p /opt/conda \
    && rm -f "${name}" "${name}.sha256" \
    && /opt/conda/bin/conda create -y -n phoebe-dev python=3.12 \
    && /opt/conda/bin/conda clean -afy

# environment modules config:
RUN mkdir -p /opt/modulefiles/tether-probe \
    && printf '#%%Module1.0\nsetenv TETHER_MODULE_LOADED 1\n' \
        > /opt/modulefiles/tether-probe/1.0 \
    && echo /opt/modulefiles >> /etc/environment-modules/modulespath

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 22
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
