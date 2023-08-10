import math
import re
import time

from flask import Blueprint, render_template, redirect, session, request, flash, url_for, jsonify, send_from_directory, \
    current_app
from flask_cors import cross_origin
from jinja2 import pass_eval_context


main_page = Blueprint("main_page", __name__, template_folder="templates")


def sanitize_input(in_str, regex=r'[^a-zA-Z0-9():/\\.,\-&?#= ]'):
    result = re.sub(regex, "", in_str)
    return result


def floatify(x, sanitize_mode=False):
    for key in x:
        if (key == "mismatch" and isinstance(x[key], dict)) or sanitize_mode:
            if isinstance(x[key], dict):
                x[key] = floatify(x[key], sanitize_mode=True)
            else:
                if not (isinstance(x[key], int) or isinstance(x[key], float)):
                    x[key] = sanitize_input(x[key])
        else:
            if isinstance(x[key], dict):
                x[key] = floatify(x[key], sanitize_mode=sanitize_mode)
            else:  # if isinstance(value, str):
                x[key] = float(x[key])
    return x


@main_page.context_processor
def utility_processor():
    def is_user_admin(user_id):
        try:
            return False
        except:
            return False

    return dict(is_user_admin=is_user_admin)


@main_page.app_template_filter()
@pass_eval_context
def to_ctime(eval_ctx, ms_time):
    try:
        return time.ctime(ms_time / 1000)
    except:
        return "NaN"


@main_page.route("/drf")
def main_index():
    print(current_app.app_context())
    return render_template('inderx.html'), 200

@main_page.route("/impressum")
def impressum():
    return render_template("impressum.html")

