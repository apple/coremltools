import os
import pathlib
import tempfile

import git
from github import Github

# get a reference to the current dir's repo (coremltools clone dir)
temp_dir = tempfile.mkdtemp()
existing_local_repo = git.Repo(pathlib.Path(__file__).parent.parent.absolute())
local_repo = existing_local_repo.clone(temp_dir)

# since this is cloned from a local repo, its origin points to the local fs.
# add a 'gitlab_build' remote we can push to.
git.remote.Remote.add(local_repo, 'gitlab_build', 'git@gitlab.com:zach_nation/coremltools.git')
current_remotes = set([remote.name for remote in local_repo.remotes])

g = Github(os.getenv('COREMLTOOLS_GITHUB_API_TOKEN'))
remote_repo = g.get_repo('apple/coremltools')
pulls = remote_repo.get_pulls(state='open', sort='created', base='main')
for pr in pulls:
    remote_name = pr.head.repo.owner.login
    remote_url = pr.head.repo.clone_url
    if not(remote_name in current_remotes):
        print('Adding remote {} with url {}'.format(remote_name, remote_url))
        git.remote.Remote.add(local_repo, remote_name, remote_url)
        current_remotes.add(remote_name)
    remote = git.remote.Remote(local_repo, remote_name)
    remote.fetch()

    # check out the PR branch
    print('Checking out branch {} from remote {}'.format(pr.head.ref, remote_name))
    branch_ref = git.refs.remote.RemoteReference(local_repo, 'refs/remotes/{}/{}'.format(remote_name, pr.head.ref))
    branch_ref.checkout()

    new_branch_name = 'PR-{}'.format(pr.number)
    print('Checking out new branch {}'.format(new_branch_name))
    local_repo.git.checkout('HEAD', b=new_branch_name)

    print('Pushing PR branch to gitlab_build remote')
    gitlab_build = git.remote.Remote(local_repo, 'gitlab_build')
    gitlab_build.push('{}:{}'.format(new_branch_name, new_branch_name))
