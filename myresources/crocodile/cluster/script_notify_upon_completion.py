

from crocodile.comms.notification import Email
import crocodile.toolbox as tb

# to be replaced
addressee = ""
speaker = ""
ssh_conn_str = ""
executed_obj = ""
job_id = ""
email_config_name = ""
to_email = ""


to_be_deleted = ["exec_times = tb.S()", "shell_script_path = tb.P()", "py_script_path = tb.P()", "res_folder = tb.P()", "error_message = ''"]
error_message = ''
exec_times = tb.S()
shell_script_path = tb.P()
py_script_path = tb.P()
res_folder = tb.P()

print(f'SENDING notification email using `{email_config_name}` email configuration ...')

sep = "\n" * 2  # SyntaxError: f-string expression part cannot include a backslash, keep it here outside fstring.
Email.send_and_close(config_name=email_config_name, to=to_email,
                     subject=f"Execution Completion Notification, job_id = {job_id}",
                     msg=f'''

Hi `{addressee}`, I'm `{speaker}`, this is a notification that I have completed running the script you sent to me.

``` {executed_obj} ```


#### Error Message:
`{error_message}`
#### Execution Times
{exec_times.print(as_config=True, return_str=True)}
#### Executed Shell Script: 
`{shell_script_path}`
#### Executed Python Script: 
`{py_script_path}`

#### Pull results using this script:
`ftprx {ssh_conn_str} {res_folder.collapseuser().as_posix()} -r`
Or, using croshell,

```python

ssh = SSH(r'{ssh_conn_str}')
ssh.copy_to_here(r'{res_folder.collapseuser().as_posix()}', r=False, zip_first=False)

```

#### Results folder contents:
```

{res_folder.search().print(return_str=True, sep=sep)}

```

'''
                     )

print(f'FINISHED sending notification email to `{to_email}`')

