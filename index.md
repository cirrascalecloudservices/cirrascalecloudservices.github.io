---
# You don't need to edit this file, it's empty on purpose.
# Edit theme's home layout instead if you wanna make some changes
# See: https://jekyllrb.com/docs/themes/#overriding-theme-defaults
layout: home
---
<div class="blog-index">
{% for post in site.posts %}
{% if post.title == 'Scalable Deep Learning at Cirrascale - Blog Series' %}
  {% assign content = post.content %}
  {% include post_detail.html %}
{% endif %}
{% endfor %}
</div>