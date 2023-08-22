# Contributing

The Core ML `.mlmodel` file format is a publicly documented specification. The Core ML Tools source code is 100% open source under the [BSD license](https://en.wikipedia.org/wiki/BSD_licenses). 

As the Core ML open source community, we welcome all contributions and ideas to grow the product. We ask that you follow the contributing guidelines and code of conduct](https://github.com/apple/coremltools/blob/master/CONTRIBUTING.md, which are typical of open source communities. See [Contribution Guidelines](https://github.com/apple/coremltools/blob/master/CONTRIBUTING.md).

You can contribute in the following ways:

- [Issues and queries](#issues-and-queries): Tell us about an issue, request a feature or enhancement, or ask a question.
- [Documentation](#documentation): Help us improve the documentation.
- [Contributions](#contributions): Add new code to improve a feature or add functionality.

```{admonition} Source code

For the source code, see the [`coremltools` GitHub repository](https://github.com/apple/coremltools).
```

## Issues and queries

We encourage you to resolve or add comments to [any open issue](https://github.com/apple/coremltools/issues) in the repository.

[**Use these templates**](https://github.com/apple/coremltools/issues/new/choose) to tell us about a bug or issue, request a feature or enhancement, or ask a question. Fill in the template as much as possible to help others in the community understand what you are saying, so that they can promptly respond. If applicable, provide the model you used when logging an issue, and any code or scripts to reproduce the issue. 

If you find a software issue, follow these steps:

1. Enter the following pip command to ensure that you are using the newest version of the `coremltools` package:
    
	```shell
	pip install coremltools --upgrade
	```

	```{admonition} Install/upgrade instructions
	For instructions on installing or upgrading Core ML Tools, see [Installing Core ML Tools](installing-coremltools).
	```

2. Check [currently open pull requests](https://github.com/apple/coremltools/pulls) in the repository to see if the issue is already being addressed. 
3. Check [open issues](https://github.com/apple/coremltools/issues) to see if the issue has already been reported. Use the **Label** dropdown menu to filter issues by a [label](#labels) such as **bug**. If an issue already exists, add a comment or thumbs-up to indicate that the others are having the same problem.  
4. Try to reproduce the problem, copy any results or errors, and paste them into your issue report. 
5. Provide useful information about your configuration, such as the OS version, coremltools version, and so on. The [template](https://github.com/apple/coremltools/issues/new/choose) walks you through this process.

```{admonition} Security issue

An ideal issue report includes a script to completely reproduce the issue along with models, sample code, or data required for the script. If you are not comfortable sharing this publicly, please [file a report with developer.apple.com](https://developer.apple.com/bug-reporting/).
```

Once you submit an issue, feature request, or question, members of the community will review it. The Core ML team will determine how to proceed with it, and add the appropriate [labels](#labels) to it.

## Documentation

Help us improve the documentation. Even if you find only a typo, don’t hesitate to report it. To make changes, use one of these methods:

- Click the **SUGGEST EDITS** button in the top right corner of the documentation page, and edit the text.
- Send a pull request as described in [Contributions](#contributions) and add the **docs** label to it (see [Labels](#labels) for details).

## Contributions

Add new functionality to the Core ML Tools repository by submitting a GitHub [pull request](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request). 

- For an example, see [this pull request](https://github.com/apple/coremltools/pull/452) for enhancing the quantization utility. 
- To see pull requests still in progress, see the [list of current pull requests](https://github.com/apple/coremltools/pulls). 
- For instructions on forking the repository and creating pull requests, see [GitHub Standard Fork & Pull Request Workflow](https://gist.github.com/Chaser324/ce0505fbed06b947d962).

Before contributing code, be sure to install the source code properly and test your code. For details, see [Building from Source](https://github.com/apple/coremltools/blob/master/BUILDING.md).

Once you submit a pull request, members of the community will review it. The Core ML team will determine how to proceed with it, and add the appropriate [labels](#labels) to it. A pull request must be approved by a Core ML team member.

## Labels

Core ML team members will add labels to your issues, requests, questions, or pull requests. For a description of each label, see the [labels page](https://github.com/apple/coremltools/labels) in the repository. An issue typically has the following types of labels:

- Status label (turquoise in color): The issue’s stage in the process. Status labels include: 
  - **triaged**: Team members have reviewed and examined the issue and assigned a release (if applicable). The issue may still be awaiting a response or investigation, or may need discussion.
  - **awaiting response**: The issue needs a response from the issue’s author.
  - **duplicate**: The issue is a duplicate. Progress will appear on a similar previously-logged issue.
  - **repro needed**: Team members need more information to reproduce the issue.
  - **investigation**: Team members are investigating the issue.
- Type of issue (red in color):
  - **bug**: Unexpected behavior that should be fixed. Use this label with issues you create using the [bug template](https://github.com/apple/coremltools/issues/new?assignees=&labels=bug&template=---bug-report.md&title=).
  - **docs**: Errors or accuracy issues in the documentation, including requests for clarification and additional information.
  - **enhancement**: An improvement to an existing feature.
  - **feature request**: Functionality that doesn’t currently exist. Use this label with issues you create using the [feature request template](https://github.com/apple/coremltools/issues/new?assignees=&labels=feature-request&template=---feature-request.md&title=).
  - **question**: Question to the team, such as a request for clarification. Use this label with issues you create using the [question template](https://github.com/apple/coremltools/issues/new?assignees=&labels=question&template=-question.md&title=).
- Framework label (orange in color)_:_ Use this label to specify the framework that this issue is appearing in. Examples include **caffe**, **onnx**, **pytorch**, **tf1.x** (TensorFlow 1), and **tf2.x / tf.keras** (TensorFlow 2 and TensorFlow Keras).
- Issues for contributors to complete: If an issue is good for a contributor to self-assign, it may include the **good first issue** label.
