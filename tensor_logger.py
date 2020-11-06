import tensorboardX
from checkpoint_manager import CheckpointManager

class Logger:

    def __init__(self, logdir, type='tensorboardX', 
                  summary=True, step=None, 
                  iterations=0, epoch=0, save_frequency=50, tag='log_'):
        self.logger = None
        self.type = type
        self.step = step
        self.checkpoint = CheckpointManager(logdir)
        self.iterations = iterations
        self.epoch = epoch
        self.save_frequency = save_frequency
        self.tag = tag

        self.summary = summary
        if summary:
            if type == 'tensorboardX':
                self.logger = tensorboardX.SummaryWriter(logdir)
            else:
                raise NotImplementedError
        else:
            self.type = 'None'

    def step_it(self):
        self.iterations += 1
    
    def save_model(self, models):
        self.checkpoint.save_checkpoint(self.epoch, self.iterations, models, self.tag)  

    def step_epoch(self, models):
        self.epoch += 1
        if self.epoch % self.save_frequency == 0:
            self.checkpoint.save_checkpoint(self.epoch, self.iterations, models, self.tag)  

    def close(self):
        if self.logger is not None:
            self.logger.close()
        self.info("Closing the Logger.")

    def add_scalar(self, tag, scalar_value, step=None):
        if self.type == 'tensorboardX':
            tag = self._transform_tag(tag)
            self.logger.add_scalar(tag, scalar_value, step)

    def add_image(self, tag, image, step=None):
        if self.type == 'tensorboardX':
            tag = self._transform_tag(tag)
            self.logger.add_image(tag, image, step)

    def add_figure(self, tag, image, step=None):
        if self.type == 'tensorboardX':
            tag = self._transform_tag(tag)
            self.logger.add_figure(tag, image, step)

    def add_table(self, tag, tbl, step=None):
        if self.type == 'tensorboardX':
            tag = self._transform_tag(tag)
            tbl_str = "<table width=\"100%\"> "
            tbl_str += "<tr> \
                     <th>Term</th> \
                     <th>Value</th> \
                     </tr>"
            for k, v in tbl.items():
                tbl_str += "<tr> \
                           <td>%s</td> \
                           <td>%s</td> \
                           </tr>" % (k, v)

            tbl_str += "</table>"
            self.logger.add_text(tag, tbl_str, step)

    def _transform_tag(self, tag):
        tag = tag + "/{self.step}" if self.step is not None else tag
        return tag

    def add_results(self, results):
        if self.type == 'tensorboardX':
            tag = self._transform_tag("Results")
            text = "<table width=\"100%\">"
            for k, res in results.items():
                text += "<tr><td>{k}</td>" + " ".join([str('<td>{x}</td>') for x in res.values()]) + "</tr>"
            text += "</table>"
            self.logger.add_text(tag, text)
