# refer: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status

ERROR_DICT = {
    "U00": "Undefined error",
    # 0-10: API related error (reserved)
    "A01": "Invalid Parameters or Values!",
    "A02": "Failed in cropping",
    "A03": "Unable to find specified document in the image",
    "A07": "Invalid image file",
    "A08": "Bad data",
    "A09": "Unable to find any image",
    "A10": "Have too many images",
    ##
    "A11": "Market service not available",
    "A12": "Failed to call linked module",
    "A13": "Object not found",
    ##
    "A14": "Type is not supported",

    # Engine (models, config) related problems
    "E01": "Config application error",
    "E02": "Application type is not supported",
    "E03": "Error loading model",
    "E04": "There's no success train task",

    # Predict problems
    "P01": "Error in prediction",
    "P02": "Model failed to predict",
    "P03": "Predict confidence too low (< threshold)",

    # Training/automation related problems
    "T01": "Not enough sample",
}


def build_predict_headers(app, billing_units=0, consumer_id=""):
    return {
        'Reader-FPTAI-BILLING': str(billing_units),
        'Reader-consumer-id': consumer_id,
        'Reader-application-id': app['application_id'],
        'Reader-type': app['type'],
        'Reader-market': app['config']['market_id'] if 'market_id' in app['config'] else ""
    }


class CustomError(Exception):
    def __init__(self,
                 status_code=500,
                 error_code=None,
                 message=None,
                 headers={}):
        super(CustomError, self).__init__(message)

        self.status_code = status_code
        if error_code is None:
            self.error_code = str(status_code)
        else:
            self.error_code = error_code
        if message is None and error_code in ERROR_DICT:
            self.message = ERROR_DICT[self.error_code]
        else:
            self.message = message
        self.headers = headers

    def log(self):
        return {
            'status_code': self.status_code,
            'error_code': self.error_code,
            'message': self.message
        }

    def __str__(self):
        return "'status_code': {}, 'error_code': {}, 'message': {}".format(
            self.status_code,
            self.error_code,
            self.message
        )


class TimeoutException(Exception):
    pass
