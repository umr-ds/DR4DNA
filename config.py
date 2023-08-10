# coding=utf-8
import os


def str2bool(txt):
    return txt is not None and txt.lower() in ['true', "'true'"]


# only needed if not using the docker webserver
try:
    docker_name = 'mesadnasim'  # os.path.dirname(os.path.realpath(__file__)).split("/")[-1]
    postgres_ip = \
        os.popen(
            'docker network inspect ' + docker_name + '_no-internet | grep "postgres" -A 4 | grep "IPv4Address"').read().split(
            '": "')[1].split("/")[0]

    redis_ip = \
        os.popen(
            'docker network inspect ' + docker_name + '_no-internet | grep "redis" -A 4 | grep "IPv4Address"').read().split(
            '": "')[1].split("/")[0]
except:
    print("Failed getting docker IPs, using fallback values!")
    postgres_ip = '172.22.0.2'  # docker network inspect dnasim_no-internet | grep "postgres" -A 4 | grep "IPv4Address"
    redis_ip = os.environ.get('REDIS_SERVER') or '172.18.0.2'  # docker network inspect dnasim_no-internet | grep "redis" -A 4 | grep "IPv4Address"

redis_password = os.environ.get('REDIS_PASSWORD') or None  # set to None if not needed


class Config(object):
    DEBUG = True
    TESTING = True
    CSRF_ENABLED = True
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SECRET_KEY = os.environ.get('SECRET_KEY')
    SECRET_VALIDATION_SALT = os.environ.get('SECRET_VALIDATION_SALT')
    SECRET_PASSWORD_RESET_VALIDATION_KEY = os.environ.get('SECRET_PASSWORD_RESET_VALIDATION_KEY')
    SECRET_ACCOUNT_DELETION_VALIDATION_KEY = os.environ.get('SECRET_ACCOUNT_DELETION_VALIDATION_KEY')


    EXCEPTION_RECV = os.environ.get('EXCEPTION_EMAIL') or None

    SQLALCHEMY_DATABASE_URI = os.environ.get(
        'SQLALCHEMY_DATABASE_URI')


class ProductionConfig(Config):
    DEBUG = False
    TESTING = False
    DEVELOPMENT = False
    MAIL_DEBUG = False


class StagingConfig(Config):
    DEVELOPMENT = True
    DEBUG = True


class DevelopmentConfig(Config):
    DEVELOPMENT = True
    DEBUG = True


class TestingConfig(Config):
    TESTING = True
