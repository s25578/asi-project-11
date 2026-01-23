FROM docker:dind

RUN apk upgrade \
  && apk update \
	&& apk add --no-cache \
		openssh-server \
		openssh-sftp-server \
		shadow \
		sudo \
        ansible \
        sshpass \
    && ssh-keygen -A

RUN apk add --no-cache python3

COPY entrypoint.sh /tmp-entrypoint.sh
RUN tr -d '\r' < tmp-entrypoint.sh > entrypoint.sh && rm tmp-entrypoint.sh
RUN chmod +x entrypoint.sh

RUN useradd -m ansible \
	&& echo 'ansible ALL=(ALL:ALL) ALL' >> /etc/sudoers

ENV USERNAME=ansible

ENTRYPOINT ["/entrypoint.sh"]
