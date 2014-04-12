import tornado.ioloop
import tornado.web

docs_path = '_build/html/'
settings = {'debug': True}
port = 2709

handlers = [(r'/(.*)', tornado.web.StaticFileHandler, {'path': docs_path}),
            ]

if __name__ == "__main__":

    try:
        application = tornado.web.Application(handlers, **settings)
        application.listen(port)
        tornado.ioloop.IOLoop.instance().start()
    except KeyboardInterrupt:
        tornado.ioloop.IOLoop.instance().stop()