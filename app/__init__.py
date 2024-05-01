import os
from flask import Flask

template_dir = os.path.abspath('app/ui/templates')

app = Flask(__name__,template_folder=template_dir,static_url_path='/static', static_folder='static')

# Import blueprints
from app.ui.views import ui_blueprint

# Register blueprints
app.register_blueprint(ui_blueprint)
