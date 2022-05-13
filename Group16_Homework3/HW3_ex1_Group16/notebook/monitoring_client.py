from simpleSubscriber import MySubscriber

test = MySubscriber("MySubscriber", "/temperature_alert", "/humidity_alert")
test.start()

while True:
    continue