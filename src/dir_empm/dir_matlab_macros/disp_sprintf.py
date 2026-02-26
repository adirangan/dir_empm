#import os; os.chdir('/data/rangan/dir_cryoem/dir_rangan_python');
import io ;

def sprintf(format_str, *format_args):
    tmp_str = io.StringIO();
    tmp_str.write(format_str % format_args);
    return(tmp_str.getvalue());

def disp(input_str):
    print(input_str);

