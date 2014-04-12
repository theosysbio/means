import tornado.ioloop
import tornado.web
import os
import re

static_path = '.'
settings = {'debug': True}
port = 2712

def _read_index_template():
    with open('index.template', 'r') as f:
        return f.read()
INDEX_TEMPLATE = _read_index_template()

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        template = INDEX_TEMPLATE
        tutorial_list = []
        for filename in os.listdir(static_path):
            if filename.endswith('.html'):
                # Remove the .html in the ned
                name_of_tutorial = filename[:-5]
                name_of_tutorial = re.sub('^\d+\.\s+', '', name_of_tutorial)
                tutorial_list.append('<li><a href="{0}">{1}</a></li>'.format(filename, name_of_tutorial))

        self.write(template.format(tutorial_files_list='\n'.join(tutorial_list)))

handlers = [(r'/$', MainHandler),
            (r'/(.*)$', tornado.web.StaticFileHandler, {'path': static_path}),
            ]

if __name__ == "__main__":

    try:
        application = tornado.web.Application(handlers, **settings)
        application.listen(port)
        tornado.ioloop.IOLoop.instance().start()
    except KeyboardInterrupt:
        tornado.ioloop.IOLoop.instance().stop()
