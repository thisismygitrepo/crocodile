

from crocodile.comms.notification import Email
import crocodile.toolbox as tb

print('SENDING notification email ...')

# to be replaced
addressee = ""
speaker = ""
executed_obj = ""
ssh_username = ""
ssh_hostname = ""
job_id = ""
email_config_name = ""
to_email = ""


to_be_deleted = ["exec_times = tb.S()", "shell_script_path = tb.P()", "py_script_path = tb.P()", "res_folder = tb.P()", "error_message = ''"]
error_message = ''
exec_times = tb.S()
shell_script_path = tb.P()
py_script_path = tb.P()
res_folder = tb.P()


sep = "\n" * 2  # SyntaxError: f-string expression part cannot include a backslash, keep it here outside fstring.
Email.send_and_close(config_name=email_config_name, to=to_email,
                     subject=f"Execution Completion Notification, job_id = {job_id}",
                     msg=f'''

Hi `{addressee}`, I'm `{speaker}`, this is a notification that I have completed running the script you sent to me.

#### Error Message:
`{error_message}`
#### Execution Times
{exec_times.print(as_config=True, return_str=True)}
#### Executed Python Function: 
``` {executed_obj} ```
#### Executed Shell Script: 
`{shell_script_path}`
#### Executed Python Script: 
`{py_script_path}`

#### Pull results using this script:
`ftprx {ssh_username}@{ssh_hostname} {res_folder.collapseuser().as_posix()} -r`
Or, using croshell,

```python

ssh = SSH(r'{ssh_username}', r'{ssh_hostname}')
ssh.copy_to_here(r'{res_folder.collapseuser().as_posix()}', r=False, zip_first=False)
```

#### Results folder contents:
```

{res_folder.search().print(return_str=True, sep=sep)}

```

'''
                     )

print('FINISHED sending notification email')

